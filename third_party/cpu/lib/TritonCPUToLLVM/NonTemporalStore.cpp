// Sets the LLVM `nontemporal` attribute on stores that write to a
// function argument the kernel never reads. The LLVM backend then emits
// `vmovntps` / `vmovntdq` / `movnti` on x86, bypassing L1/L2 cache on
// writes and freeing cache bandwidth for the data the kernel actually
// re-reads.
//
// Empirical motivation (on i7-13700K, gcc-compiled C reference):
//   - vec_add at N >= 256K elements: non-temporal store wins 1.4-1.6x
//     over plain store (frees L2 bandwidth that would otherwise be spent
//     evicting clean lines to make room for the writes).
//   - At small N (<= 64K), non-temporal is slightly slower because the
//     write-combining buffer flush overhead exceeds the cache-pressure
//     it avoids.
//
// Scope of this pass:
//   - Only marks STORES (loads keep their cache behaviour).
//   - Only marks stores whose address ultimately comes from a function
//     pointer argument that has no `llvm.load` / `llvm.intr.masked.load`
//     reader anywhere in the function.
//   - Default OFF; enable with `TRITON_CPU_NT_STORE=1`. Default-off
//     because Triton-CPU is often used in chained workloads (RMSNorm ->
//     matmul -> ...) where the next kernel re-reads the previous output
//     and we'd LOSE by evicting that data from cache.
//
// What this pass does NOT try to handle:
//   - Per-tensor metadata about whether the next kernel will re-read
//     (would require cross-kernel analysis).
//   - Cache-size aware gating (we always mark write-only stores when
//     the env var is on; pick your kernels).
//   - Streaming loads (`vmovntdqa` etc.). Loads are left untouched.
#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdlib>
#include <string>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_NONTEMPORALSTORE
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// Walk uses transitively through aliasing ops AND through MLIR LLVM-dialect
// struct (insert/extract) wrapping. Returns true if any reachable use is a
// load-style consumer of our pointer.
//
// Triton-CPU wraps each pointer in an LLVM struct (memref descriptor):
//   %d = llvm.insertvalue %ptr, %d_in[1] : !llvm.struct<...>
// so we must follow the struct field across insertvalue and extractvalue
// with matching position before we can see the downstream load via GEP.
//
// `posStack` tracks where in a nested struct our pointer currently lives
// (empty = the value IS the raw pointer). InsertValueOp(value=v, position=p)
// pushes p; ExtractValueOp(container=v, position=p) peels p off the front.
bool hasReader(Value v, ArrayRef<int64_t> posStack,
               llvm::DenseSet<std::pair<void *, uint64_t>> &visited) {
  uint64_t posKey = 0;
  for (int64_t p : posStack)
    posKey = posKey * 1315423911u + static_cast<uint64_t>(p);
  if (!visited.insert({v.getAsOpaquePointer(), posKey}).second)
    return false;

  for (Operation *user : v.getUsers()) {
    // When v is the raw pointer (empty stack), check pointer-consuming ops.
    if (posStack.empty()) {
      if (isa<LLVM::LoadOp>(user))
        return true;
      StringRef opName = user->getName().getStringRef();
      if (opName.contains("masked.load") || opName.contains("masked.gather"))
        return true;
      if (opName.contains("memcpy") || opName.contains("memmove"))
        return true;
      if (isa<LLVM::CallOp>(user))
        return true;
      if (isa<LLVM::GEPOp, LLVM::BitcastOp, LLVM::AddrSpaceCastOp>(user)) {
        for (Value res : user->getResults())
          if (hasReader(res, posStack, visited))
            return true;
        continue;
      }
      // Pointer stored AS DATA (as the value operand) escapes via memory.
      if (auto st = dyn_cast<LLVM::StoreOp>(user))
        if (st.getValue() == v)
          return true;
    }

    // Struct embedding: handle both "v is the inserted value" and
    // "v is the container being modified" cases.
    if (auto iv = dyn_cast<LLVM::InsertValueOp>(user)) {
      if (iv.getValue() == v) {
        SmallVector<int64_t> newStack(iv.getPosition().begin(),
                                      iv.getPosition().end());
        newStack.append(posStack.begin(), posStack.end());
        if (hasReader(iv.getResult(), newStack, visited))
          return true;
      } else if (iv.getContainer() == v) {
        // Field at another position; ours survives at the existing offset.
        if (hasReader(iv.getResult(), posStack, visited))
          return true;
      }
      continue;
    }

    if (auto ev = dyn_cast<LLVM::ExtractValueOp>(user)) {
      if (ev.getContainer() != v)
        continue;
      ArrayRef<int64_t> evPos = ev.getPosition();
      if (evPos.size() > posStack.size())
        continue;
      bool match = true;
      for (size_t i = 0; i < evPos.size(); ++i) {
        if (evPos[i] != posStack[i]) {
          match = false;
          break;
        }
      }
      if (!match)
        continue;
      ArrayRef<int64_t> rest = posStack.drop_front(evPos.size());
      if (hasReader(ev.getResult(), rest, visited))
        return true;
      continue;
    }
  }
  return false;
}

// Walk an address operand back through GEPs / casts / memref-descriptor
// extract+insert chains to the originating SSA value. If that's a function
// block argument we can match it.
//
// Triton-CPU wraps each pointer in an LLVM struct of the form
//   !llvm.struct<(allocated_ptr, aligned_ptr, offset, sizes, strides)>
// so the actual `vector.store` address arrives as
//   %aligned = llvm.extractvalue %desc[1]
//   %addr    = llvm.getelementptr %aligned[%offs]
//   llvm.store %val, %addr
// where %desc comes from a chain of `llvm.insertvalue` ops constructed at
// function entry from the raw function argument. To trace through this,
// the walker follows extractvalue back through matching insertvalue for
// the same field index.
Value traceAddrRoot(Value v) {
  llvm::DenseSet<Value> visited;
  while (visited.insert(v).second) {
    Operation *def = v.getDefiningOp();
    if (!def) // BlockArgument
      return v;
    if (auto gep = dyn_cast<LLVM::GEPOp>(def)) {
      v = gep.getBase();
      continue;
    }
    if (auto bc = dyn_cast<LLVM::BitcastOp>(def)) {
      v = bc.getArg();
      continue;
    }
    if (auto as = dyn_cast<LLVM::AddrSpaceCastOp>(def)) {
      v = as.getArg();
      continue;
    }
    if (auto ev = dyn_cast<LLVM::ExtractValueOp>(def)) {
      // Walk container value back through insertvalue chain for matching idx.
      ArrayRef<int64_t> wantPos = ev.getPosition();
      Value cur = ev.getContainer();
      Value found;
      llvm::DenseSet<Value> chainVisited;
      while (chainVisited.insert(cur).second) {
        auto iv = cur.getDefiningOp<LLVM::InsertValueOp>();
        if (!iv)
          break;
        if (iv.getPosition() == wantPos) {
          found = iv.getValue();
          break;
        }
        cur = iv.getContainer();
      }
      if (!found)
        return v; // can't trace, give up
      v = found;
      continue;
    }
    return v;
  }
  return v;
}

bool ntStoreEnabled() {
  const char *e = std::getenv("TRITON_CPU_NT_STORE");
  return e && std::string(e) != "0";
}

struct NonTemporalStore
    : public triton::cpu::impl::NonTemporalStoreBase<NonTemporalStore> {
  using NonTemporalStoreBase::NonTemporalStoreBase;
  NonTemporalStore() = default;

  void runOnOperation() override {
    if (!ntStoreEnabled())
      return;

    ModuleOp mod = getOperation();
    mod.walk([&](LLVM::LLVMFuncOp fn) {
      if (fn.isExternal())
        return;

      // Identify pointer arguments that the kernel never reads from.
      llvm::DenseSet<BlockArgument> writeOnlyArgs;
      for (BlockArgument arg : fn.getArguments()) {
        if (!isa<LLVM::LLVMPointerType>(arg.getType()))
          continue;
        llvm::DenseSet<std::pair<void *, uint64_t>> visited;
        if (!hasReader(arg, /*posStack=*/{}, visited))
          writeOnlyArgs.insert(arg);
      }

      // For each store, mark non-temporal if its address derives from a
      // write-only function argument. Skip scalar stores (NT cost only
      // pays off for cache-line-sized writes).
      fn.walk([&](LLVM::StoreOp store) {
        if (store.getNontemporal())
          return;
        Type valTy = store.getValue().getType();
        if (!isa<VectorType>(valTy) && !isa<LLVM::LLVMArrayType>(valTy))
          return;
        Value root = traceAddrRoot(store.getAddr());
        auto arg = dyn_cast<BlockArgument>(root);
        if (!arg || !writeOnlyArgs.contains(arg))
          return;
        store.setNontemporal(true);
        // x86 vmovntps requires 32-byte alignment; otherwise LLVM
        // scalarizes the vector NT store into per-64-bit movnti.
        // Torch tensor allocator returns >=64-byte aligned buffers so
        // bumping the alignment annotation is safe.
        unsigned curAlign = store.getAlignment().value_or(0);
        if (curAlign < 32)
          store.setAlignment(32);
      });
    });
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createNonTemporalStorePass() {
  return std::make_unique<NonTemporalStore>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir

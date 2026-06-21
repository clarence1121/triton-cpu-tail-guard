from __future__ import annotations
import hashlib
import json
from .._C.libtriton import get_cache_invalidating_env_vars, ir
from ..backends import backends
from ..backends.compiler import Language
from ..backends.compiler import BaseBackend, GPUTarget
from .. import __version__, knobs
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager, get_cache_key
from ..runtime.driver import driver
from ..tools.disasm import get_sass
from pathlib import Path
import re
import functools
import os
import time
import copy

# - ^\s*tt\.func\s+ : match the start of the string, any leading whitespace, the keyword func,
#    and any following whitespace
# - (public\s+)? : optionally match the keyword public and any following whitespace
# - (@\w+) : match an @ symbol followed by one or more word characters
#   (letters, digits, or underscores), and capture it as group 1 (the function name)
# - (\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\)) : match a pair of parentheses enclosing
#   zero or more arguments separated by commas, and capture it as group 2 (the argument list)
# - (attributes \{[\S\s]+\})? : optionally match attributes enclosed in braces and capture it as group 3
ptx_prototype_pattern = r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(\w+)\s*\(([^)]*)\)"
prototype_pattern = {
    "ptx": ptx_prototype_pattern,
}

ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ptx": ptx_arg_type_pattern,
}


def convert_type_repr(x):
    # Currently we only capture the pointer type and assume the pointer is on global memory.
    # TODO: Capture and support shared memory space
    match = re.search(r'!tt\.ptr<([^,]+)', x)
    tma = re.search(r'tt.nv_tma_desc = 1', x)
    if tma is not None:
        return 'nvTmaDesc'
    x = re.sub(r' {[^}]+}', '', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


class ASTSource:

    def __init__(self, fn, signature, constexprs=None, attrs=None, cpu_tail_variant=None,
                 cpu_tail_2d_variant=None,
                 cpu_region_spec=None, cpu_region_pid_vars=None, cpu_region_all_specs=None) -> None:
        self.fn = fn
        self.language = Language.TRITON
        self.ext = "ttir"
        self.name = fn.__name__
        self.signature = signature
        self.cpu_tail_variant = cpu_tail_variant
        self.cpu_tail_2d_variant = cpu_tail_2d_variant
        self.cpu_region_spec = cpu_region_spec          # RegionSpec | None
        self.cpu_region_pid_vars = cpu_region_pid_vars  # list[str] | None
        self.cpu_region_all_specs = cpu_region_all_specs or []  # list[RegionSpec]
        self.constants = dict()
        if constexprs is not None:
            for k, v in constexprs.items():
                k = (fn.arg_names.index(k), ) if isinstance(k, str) else k
                assert isinstance(k, tuple)
                self.constants[k] = v
        self.attrs = attrs or dict()
        for k in self.signature.keys():
            if not isinstance(k, str):
                raise TypeError("Signature keys must be string")
        self.cpu_tail_guard_diagnostic = None
        self.cpu_pid_region_diagnostic = None

    def hash(self):
        sorted_sig = [v for k, v in sorted(self.signature.items())]
        get_key = lambda x: x.cache_key if hasattr(x, 'cache_key') else str(x)
        constants_key = '-'.join([get_key(v) for k, v in sorted(self.constants.items())])
        cpu_tail_guard_key = os.getenv("TRITON_CPU_TAIL_GUARD", "1")
        cpu_pid_region_key = os.getenv("TRITON_CPU_PID_REGION", "0")
        cpu_region_key = self.cpu_region_spec.cache_key() if self.cpu_region_spec is not None else "none"
        key = (f"{self.fn.cache_key}-{str(self.attrs)}-{sorted_sig}-{constants_key}"
               f"-{cpu_tail_guard_key}-{self.cpu_tail_variant}-{self.cpu_tail_2d_variant}"
               f"-{cpu_pid_region_key}-{cpu_region_key}")
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
        from .code_generator import ast_to_ttir
        self._parsed_tree = None
        if target.backend == "cpu" and getattr(options, "enable_tail_guard", False):
            from .cpu_tail import (analyze_cpu_tail, make_cpu_tail_main_variant,
                                   analyze_cpu_tail_2d, make_cpu_tail_2d_variant)
            if self.cpu_tail_2d_variant is not None:
                self._parsed_tree, self.cpu_tail_guard_diagnostic = make_cpu_tail_2d_variant(
                    self.cpu_tail_2d_variant, self.fn, self)
            elif self.cpu_tail_variant == "main":
                self._parsed_tree, self.cpu_tail_guard_diagnostic = make_cpu_tail_main_variant(self.fn, self)
            else:
                # Diagnostic only (no rewrite): try 2D first, fall back to 1D
                a2d = analyze_cpu_tail_2d(self.fn, self)
                if a2d.diagnostic.matched:
                    self.cpu_tail_guard_diagnostic = a2d.diagnostic
                else:
                    self.cpu_tail_guard_diagnostic = analyze_cpu_tail(self.fn, self).diagnostic
            debug_tail_guard = os.getenv("TRITON_CPU_TAIL_GUARD_DEBUG", "0") == "1" or getattr(options, "debug", False)
            if debug_tail_guard and self.cpu_tail_guard_diagnostic is not None:
                diag = self.cpu_tail_guard_diagnostic
                status = "matched" if diag.matched else f"rejected: {diag.reason}"
                variant = f" ({self.cpu_tail_variant})" if self.cpu_tail_variant else ""
                print(f"triton-cpu tail_guard{variant}: {self.name}: {status}")
        if target.backend == "cpu" and self.cpu_region_spec is not None:
            from .cpu_pid_region import make_region_variant
            self._parsed_tree, _ok = make_region_variant(
                self.fn, self, self.cpu_region_spec,
                self.cpu_region_all_specs, self.cpu_region_pid_vars or [])
            debug_region = (os.getenv("TRITON_CPU_PID_REGION_DEBUG", "0") == "1"
                            or getattr(options, "debug", False))
            if debug_region:
                print(f"triton-cpu pid_region ({self.cpu_region_spec.name}): {self.name}")
        return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                           module_map=module_map)

    def parse_options(self):
        return dict()

    def parse(self):
        if getattr(self, "_parsed_tree", None) is not None:
            return self._parsed_tree
        return self.fn.parse()


class IRSource:

    def __init__(self, path, context, backend):
        self.path = path
        path = Path(path)
        self.ext = path.suffix[1:]
        self.language = Language.TRITON
        self.src = path.read_text()
        ir.load_dialects(context)
        backend.load_dialects(context)

        # We don't have a easy-to-use PTX parser that we can use, so keep that regex for now.
        # TODO - replace with a proper parser
        if self.ext == "ptx":
            match = re.search(prototype_pattern[self.ext], self.src, re.MULTILINE)
            self.name = match.group(1)
            signature = match.group(2)
            types = re.findall(arg_type_pattern[self.ext], signature)
            self.signature = {k: convert_type_repr(ty) for k, ty in enumerate(types)}
        else:
            self.module = ir.parse_mlir_module(self.path, context)
            fn_name = self.module.get_entry_func_name()
            self.name = "@" + fn_name
            funcOp = self.module.get_function(fn_name)
            func_ty = self.module.get_function_signature(funcOp)
            self.signature = {k: ty for k, ty in enumerate(func_ty)}

    def hash(self):
        return hashlib.sha256(self.src.encode("utf-8")).hexdigest()

    def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
        self.module.context = context
        return self.module

    def parse_options(self):
        if self.ext == "ttgir":
            num_warps = self.module.get_int_attr("ttg.num-warps")
            assert num_warps is not None, "Unable to parse ttg.num-warps attribute"
            options = {'num_warps': num_warps}
            num_ctas = self.module.get_int_attr("ttg.num-ctas")
            if num_ctas is not None:
                options['num_ctas'] = num_ctas
            return options
        return dict()


@functools.lru_cache()
def max_shared_mem(device):
    return driver.active.utils.get_device_properties(device)["max_shared_mem"]


def parse(full_name, ext, context):
    if ext == "ttir" or ext == "ttgir":
        module = ir.parse_mlir_module(full_name, context)
        module.context = context
        return module
    if ext == "llir" or ext == "ptx" or ext == "amdgcn":
        return Path(full_name).read_text()
    if ext == "cubin" or ext == "hsaco":
        return Path(full_name).read_bytes()


def filter_traceback(e: BaseException):
    """
    Removes code_generator.py and related files from tracebacks.

    These are uninteresting to the user -- "just show me *my* code!"
    """
    if knobs.compilation.front_end_debugging:
        return

    if e.__cause__ is not None:
        filter_traceback(e.__cause__)
    if e.__context__ is not None:
        filter_traceback(e.__context__)

    # If a user has a file that matches one of these, they're out of luck.
    BAD_FILES = [
        "/triton/compiler/code_generator.py",
        "/ast.py",
    ]
    BAD_FILES = [bad_file.replace("/", os.sep) for bad_file in BAD_FILES]

    tb = e.__traceback__
    frames = []
    while tb is not None:
        if not any(f for f in BAD_FILES if tb.tb_frame.f_code.co_filename.endswith(f)):
            frames.append(tb)
        tb = tb.tb_next

    for (cur_frame, next_frame) in zip(frames, frames[1:]):
        cur_frame.tb_next = next_frame

    if not frames:
        e.__traceback__ = None
    else:
        frames[-1].tb_next = None
        e.__traceback__ = frames[0]


class CompileTimer:

    def __init__(self) -> None:
        self.start: float = time.time()
        self.ir_initialization_end: float | None = None
        self.lowering_stage_ends: list[tuple[str, float]] = []
        self.store_results_end: float | None = None

    def finished_ir_initialization(self) -> None:
        self.ir_initialization_end = time.time()

    def stage_finished(self, stage_name: str) -> None:
        self.lowering_stage_ends.append((stage_name, time.time()))

    def end(self) -> knobs.CompileTimes:
        timestamp = time.time()
        if self.ir_initialization_end is None:
            self.ir_initialization_end = timestamp
        else:
            self.store_results_end = timestamp

        def delta(start: float, end: float | None) -> int:
            if end is None:
                return 0
            return int((end - start) * 1000000)

        lowering_stage_durations = []
        stage_start = self.ir_initialization_end
        for stage_name, stage_end in self.lowering_stage_ends:
            lowering_stage_durations.append((stage_name, delta(stage_start, stage_end)))
            stage_start = stage_end

        return knobs.CompileTimes(
            ir_initialization=delta(self.start, self.ir_initialization_end),
            lowering_stages=lowering_stage_durations,
            store_results=delta(stage_start, self.store_results_end),
        )


def compile(src, target=None, options=None, _env_vars=None):
    compilation_listener = knobs.compilation.listener
    if compilation_listener:
        timer = CompileTimer()

    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    # create backend
    if ir_source:
        assert isinstance(src, str), "source must be either AST or a filepath"
        context = ir.context()
        src = IRSource(src, context, backend)

    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))
    # create cache manager
    env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
    key = get_cache_key(src, backend, options, env_vars=env_vars)
    if knobs.runtime.add_stages_inspection_hook is not None:
        inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
        key += inspect_stages_key
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = knobs.compilation.override
    enable_ir_dump = knobs.compilation.dump_ir
    store_only_binary = knobs.compilation.store_binary_only
    fn_override_manager = get_override_manager(src.hash()) if enable_override else None
    fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = src.name[:150]
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = knobs.compilation.always_compile
    if not always_compile and metadata_path is not None:
        # cache hit!
        res = CompiledKernel(src, metadata_group, hash)
        if compilation_listener:
            compilation_listener(
                src=src,
                metadata=res.metadata._asdict(),
                metadata_group=metadata_group,
                times=timer.end(),
                cache_hit=True,
            )
        return res

    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = __version__
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(stages, options, src.language)
    first_stage = list(stages.keys()).index(src.ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1

    # For IRSource, we have already grabbed the context + called both
    # ir.load_dialects and backend.load_dialects.
    if not isinstance(src, IRSource):
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    try:
        module = src.make_ir(target, options, codegen_fns, module_map, context)
    except Exception as e:
        filter_traceback(e)
        raise
    if isinstance(src, ASTSource) and src.cpu_tail_guard_diagnostic is not None:
        metadata["cpu_tail_guard"] = src.cpu_tail_guard_diagnostic.asdict()
    if isinstance(src, ASTSource) and src.cpu_pid_region_diagnostic is not None:
        metadata["cpu_pid_region"] = src.cpu_pid_region_diagnostic.asdict()

    if ir_source:
        ir_filename = f"{file_name}.{src.ext}"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)
    else:
        ir_filename = f"{file_name}.source"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

    use_ir_loc = knobs.compilation.use_ir_loc
    if ir_source and use_ir_loc:
        module.create_location_snapshot(src.path)
        print(f"Creating new locations for {src.path}")

    if compilation_listener:
        timer.finished_ir_initialization()
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        if fn_override_manager is None:
            # Users can override kernels at scale by setting `ir_override` in autotune config
            # without TRITON_KERNEL_OVERRIDE
            if (ir_override := metadata.get("ir_override", None)) and ir_override.endswith(f".{ext}"):
                next_module = parse(ir_override, ext, context)
        elif full_name := fn_override_manager.get_file(ir_filename):
            print(f"\nOverriding kernel with file {full_name}")
            next_module = parse(full_name, ext, context)
        # If TRITON_STORE_BINARY_ONLY is 1, only store cubin/hsaco/json
        if (not store_only_binary) or (ext in ("cubin", "hsaco", "json")):
            metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
            if ext == "cubin":
                sass = get_sass(next_module)
                fn_dump_manager.put(sass, file_name + ".sass")
        # use an env variable to parse ir from file
        if use_ir_loc == ext:
            ir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ir_full_name)
            print(f"Creating new locations for {ir_full_name}")
        module = next_module
        if compilation_listener:
            timer.stage_finished(ext)
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)

    # notify any listener
    if compilation_listener:
        compilation_listener(src=src, metadata=metadata, metadata_group=metadata_group, times=timer.end(),
                             cache_hit=False)
    # return handle to compiled kernel
    return CompiledKernel(src, metadata_group, hash)


def make_backend(target: GPUTarget) -> BaseBackend:
    actives = [x.compiler for x in backends.values() if x.compiler.supports_target(target)]
    if len(actives) != 1:
        raise RuntimeError(
            f"{len(actives)} compatible backends for target ({target.backend}) ({actives}). There should only be one.")
    return actives[0](target)


class LazyDict:

    def __init__(self, data):
        self.data = data
        self.extras = []

    def get(self):
        for func, args in self.extras:
            self.data = self.data | func(*args)
        self.extras.clear()
        return self.data

    def add(self, func, args):
        self.extras.append((func, args))


class AsmDict(dict):

    def __missing__(self, key):

        if key == "sass":
            value = get_sass(self["cubin"])
        else:
            raise KeyError("Unknown key: '%s'" % key)

        self[key] = value
        return value


def _raise_error(err, *args, **kwargs):
    raise copy.deepcopy(err)


class CompiledKernel:

    def __init__(self, src, metadata_group, hash):
        from collections import namedtuple
        metadata_path = next((Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        metadata = json.loads(metadata_path.read_text())
        # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
        target = metadata['target']
        metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)
        backend = make_backend(self.metadata.target)
        self.packed_metadata = backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name
        # stores the text of each level of IR that was generated during compilation
        asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = backend.binary_ext
        self.asm = AsmDict({
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] == binary_ext else file.read_text()
            for file in asm_files
        })
        self.metadata_group = metadata_group
        self.kernel = self.asm[binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None
        self._run = None

    def __del__(self):

        if self.module is not None:
            if knobs.runtime.kernel_unload_hook is not None:
                knobs.runtime.kernel_unload_hook(self.module, self.function, self.name, self.metadata_group, self.hash)

            driver.active.utils.unload_module(self.module)
            self.module = None

    def _init_handles(self):
        if self.module is not None:
            return

        def raise_(err):
            # clone the exception object so that the one saved in the closure
            # of the partial function below doesn't get assigned a stack trace
            # after the subsequent raise. otherwise, the CompiledKernel instance
            # saved in the (global) kernel cache will keep references to all the
            # locals in the traceback via the exception instance in the closure.
            cloned_err = copy.deepcopy(err)
            self._run = functools.partial(_raise_error, cloned_err)
            raise err

        device = driver.active.get_current_device()
        # create launcher
        self._run = driver.active.launcher_cls(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = max_shared_mem(device)
        if self.metadata.shared > max_shared:
            raise_(OutOfResources(self.metadata.shared, max_shared, "shared memory"))
        if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
            # Use blackwell max tmem size for now, this should be moved in device properties
            max_tmem_size = 512  # tmem size in number of columns
            if self.metadata.tmem_size > max_tmem_size:
                raise_(OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory"))
        if knobs.runtime.kernel_load_start_hook is not None:
            knobs.runtime.kernel_load_start_hook(self.module, self.function, self.name, self.metadata_group, self.hash)
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device)
        warp_size = driver.active.get_current_target().warp_size
        if self.metadata.num_warps * warp_size > self.n_max_threads:
            raise_(OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads"))
        if knobs.runtime.kernel_load_end_hook is not None:
            knobs.runtime.kernel_load_end_hook(self.module, self.function, self.name, self.metadata_group, self.hash)

    @property
    def run(self):
        if self._run is None:
            self._init_handles()
        return self._run

    def run_with_offset(self, start_x, grid_0, grid_1, grid_2, stream, launch_metadata, launch_enter_hook,
                        launch_exit_hook, *args):
        self._init_handles()
        self._run.launch_offset(start_x, grid_0, grid_1, grid_2, stream, self.function, self.packed_metadata,
                                launch_metadata, launch_enter_hook, launch_exit_hook, *args)

    def run_region_2d(self, op, grid_0, grid_1, grid_2, stream, launch_metadata, launch_enter_hook,
                      launch_exit_hook, *args):
        self._init_handles()
        self._run.launch_region_2d(op, grid_0, grid_1, grid_2, stream, self.function, self.packed_metadata,
                                   launch_metadata, launch_enter_hook, launch_exit_hook, *args)

    def run_lower_tri_2d(self, grid_0, grid_1, grid_2, stream, launch_metadata, launch_enter_hook, launch_exit_hook,
                         *args):
        self.run_region_2d(0, grid_0, grid_1, grid_2, stream, launch_metadata, launch_enter_hook,
                           launch_exit_hook, *args)

    def run_diagonal_2d(self, grid_0, grid_1, grid_2, stream, launch_metadata, launch_enter_hook, launch_exit_hook,
                        *args):
        self.run_region_2d(2, grid_0, grid_1, grid_2, stream, launch_metadata, launch_enter_hook,
                           launch_exit_hook, *args)

    def launch_metadata(self, grid, stream, *args):
        if knobs.runtime.launch_enter_hook is None:
            return None
        self._init_handles()
        ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
        if not isinstance(self.src, ASTSource) or self.src.fn.launch_metadata is None:
            return ret
        arg_dict = {name: arg for name, arg in zip(self.src.fn.arg_names, args)}
        ret.add(self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))
        return ret

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                device = driver.active.get_current_device()
                stream = driver.active.get_current_stream(device)
            launch_metadata = self.launch_metadata(grid, stream, *args)
            self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata, launch_metadata,
                     knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)

        return runner


class DualPathCompiledKernel:

    def __init__(self, main_kernel, tail_kernel, diagnostic):
        self.main_kernel = main_kernel
        self.tail_kernel = tail_kernel
        self.metadata = tail_kernel.metadata
        self.packed_metadata = tail_kernel.packed_metadata
        self.src = tail_kernel.src
        self.hash = tail_kernel.hash
        self.name = tail_kernel.name
        self.asm = tail_kernel.asm
        self.main_asm = main_kernel.asm
        self.metadata_group = tail_kernel.metadata_group
        self.kernel = tail_kernel.kernel
        self.module = None
        self.function = None
        self._diagnostic = diagnostic
        if not isinstance(self.src, ASTSource):
            self._n_arg_idx = -1
            self._block_arg_idx = -1
        else:
            self._n_arg_idx = self.src.fn.arg_names.index(diagnostic.n_name)
            self._block_arg_idx = self.src.fn.arg_names.index(diagnostic.block_name)

    def run(self, grid_0, grid_1, grid_2, stream, function, packed_metadata, launch_metadata, launch_enter_hook,
            launch_exit_hook, *args):
        if self._n_arg_idx < 0 or self._block_arg_idx < 0:
            return self.tail_kernel.run(grid_0, grid_1, grid_2, stream, self.tail_kernel.function,
                                        self.tail_kernel.packed_metadata, launch_metadata, launch_enter_hook,
                                        launch_exit_hook, *args)
        self.main_kernel._init_handles()
        self.tail_kernel._init_handles()
        self.tail_kernel._run.launch_dual(grid_0, grid_1, grid_2, stream, self.main_kernel.function,
                                          self.main_kernel.packed_metadata, self.tail_kernel.function,
                                          self.tail_kernel.packed_metadata, launch_metadata, launch_enter_hook,
                                          launch_exit_hook, self._n_arg_idx, self._block_arg_idx, *args)

    def launch_metadata(self, grid, stream, *args):
        return self.tail_kernel.launch_metadata(grid, stream, *args)


class QuadPathCompiledKernel:
    """2-D tail guard: main / m_tail / n_tail / corner compiled variants."""

    def __init__(self, main_kernel, m_tail_kernel, n_tail_kernel, corner_kernel, diagnostic):
        self.main_kernel = main_kernel
        self.m_tail_kernel = m_tail_kernel
        self.n_tail_kernel = n_tail_kernel
        self.corner_kernel = corner_kernel
        self.metadata = corner_kernel.metadata
        self.packed_metadata = corner_kernel.packed_metadata
        self.src = corner_kernel.src
        self.hash = corner_kernel.hash
        self.name = corner_kernel.name
        self.asm = corner_kernel.asm
        self.metadata_group = corner_kernel.metadata_group
        self.kernel = corner_kernel.kernel
        self.module = None
        self.function = None
        self._diagnostic = diagnostic
        if not isinstance(self.src, ASTSource):
            self._m_n_idx = self._m_block_idx = self._n_n_idx = self._n_block_idx = -1
        else:
            names = self.src.fn.arg_names
            self._m_n_idx = names.index(diagnostic.m_n_name)
            self._m_block_idx = names.index(diagnostic.m_block_name)
            self._n_n_idx = names.index(diagnostic.n_n_name)
            self._n_block_idx = names.index(diagnostic.n_block_name)

    def run(self, grid_0, grid_1, grid_2, stream, function, packed_metadata, launch_metadata,
            launch_enter_hook, launch_exit_hook, *args):
        if self._m_n_idx < 0:
            return self.corner_kernel.run(grid_0, grid_1, grid_2, stream,
                                          self.corner_kernel.function, self.corner_kernel.packed_metadata,
                                          launch_metadata, launch_enter_hook, launch_exit_hook, *args)
        self.main_kernel._init_handles()
        self.m_tail_kernel._init_handles()
        self.n_tail_kernel._init_handles()
        self.corner_kernel._init_handles()
        self.corner_kernel._run.launch_quad(
            grid_0, grid_1, grid_2, stream,
            self.main_kernel.function, self.main_kernel.packed_metadata,
            self.m_tail_kernel.function, self.m_tail_kernel.packed_metadata,
            self.n_tail_kernel.function, self.n_tail_kernel.packed_metadata,
            self.corner_kernel.function, self.corner_kernel.packed_metadata,
            launch_metadata, launch_enter_hook, launch_exit_hook,
            self._m_n_idx, self._m_block_idx, self._n_n_idx, self._n_block_idx,
            *args)

    def launch_metadata(self, grid, stream, *args):
        return self.corner_kernel.launch_metadata(grid, stream, *args)


# Maps RegionSpec.launcher name → (CompiledKernel method, op_code)
# Op codes for run_omp_region_2d (pidY OP pidX, i.e. pid_axis1 OP pid_axis0):
#   0=LT(y<x)  1=LE(y<=x)  2=EQ(y==x)  3=GE(y>=x)  4=GT(y>x)  5=NE(y!=x)
LAUNCHER_METHOD_MAP = {
    # Canonical names
    "region_2d_lt": ("run_region_2d", 0),
    "region_2d_le": ("run_region_2d", 1),
    "region_2d_eq": ("run_region_2d", 2),
    "region_2d_ge": ("run_region_2d", 3),
    "region_2d_gt": ("run_region_2d", 4),
    "region_2d_ne": ("run_region_2d", 5),
    # Backward-compat aliases (old names still accepted)
    "lower_tri_2d": ("run_region_2d", 0),
    "diagonal_2d":  ("run_region_2d", 2),
}


class RegionCompiledKernel:
    """Multi-path compiled kernel for generic 2D pid-region dispatch.

    Accepts a list of (CompiledKernel, launcher_name) pairs, one per active
    region.  At run time each kernel is dispatched via its C-level launcher.
    """

    def __init__(self, region_kernels: list):
        # region_kernels: list of (CompiledKernel, launcher_name) ordered by dispatch priority
        self._region_kernels = region_kernels
        primary = region_kernels[-1][0]
        self.metadata = primary.metadata
        self.packed_metadata = primary.packed_metadata
        self.src = primary.src
        self.hash = primary.hash
        self.name = primary.name
        self.asm = primary.asm
        self.metadata_group = primary.metadata_group
        self.kernel = primary.kernel
        self.module = None
        self.function = None
        # Per-region asm for debugging, keyed by launcher name
        self.region_asm = {launcher: k.asm for k, launcher in region_kernels}

    def run(self, grid_0, grid_1, grid_2, stream, function, packed_metadata, launch_metadata,
            launch_enter_hook, launch_exit_hook, *args):
        for kernel, launcher in self._region_kernels:
            method_name, op = LAUNCHER_METHOD_MAP[launcher]
            getattr(kernel, method_name)(op, grid_0, grid_1, grid_2, stream, launch_metadata,
                                        launch_enter_hook, launch_exit_hook, *args)

    def launch_metadata(self, grid, stream, *args):
        return self._region_kernels[-1][0].launch_metadata(grid, stream, *args)

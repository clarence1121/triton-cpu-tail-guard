import os
import hashlib
import importlib
import importlib.resources
import tempfile
import time

import triton
import triton._C
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

from pathlib import Path
from triton._C.libtriton import llvm

_dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")
# for locating libTritonCPURuntime
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()

include_dirs = []
library_dirs = [_triton_C_dir]
libraries = ["stdc++"]
ccflags = []

# Skip non-existent paths
sys_include_dir = os.path.join(_dirname, "include")
if os.path.exists(sys_include_dir):
    include_dirs.append(sys_include_dir)

sys_lib_dir = os.path.join(_dirname, "lib")
if os.path.exists(sys_lib_dir):
    library_dirs.append(sys_lib_dir)


def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries, ccflags)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
            f.write(kernel)
            f.flush()
            import ctypes
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 0, 0, 0)

    def unload_module(self, mod):
        # TODO: implement
        pass

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr"):
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = "iiiOKOOOO" + args_format
    offset_format = "iiiiOKOOOO" + args_format

    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}

    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")

    arg_ptrs_list = ', '.join(f"&arg{i}" for i in signature.keys())
    kernel_fn_args = [i for i, ty in signature.items() if i not in constants and ty != "constexpr"]
    signature_without_constexprs = {i: ty for i, ty in signature.items() if ty != "constexpr"}
    kernel_fn_args_list = ', '.join(f"arg{i}" for i in kernel_fn_args)
    kernel_fn_arg_types = ', '.join([f"{ty_to_cpp(signature[i])}" for i in kernel_fn_args] + ["uint32_t"] * 6)

    # generate glue code
    src = f"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include <optional>
#include <stdio.h>
#include <string>
#include <memory>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

inline bool getBoolEnv(const std::string &env) {{
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) {{ return std::tolower(c); }});
  return str == "on" || str == "true" || str == "1";
}}

inline std::optional<int64_t> getIntEnv(const std::string &env) {{
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return std::nullopt;

  char *endptr;
  long int result = std::strtol(cstr, &endptr, 10);
  if (endptr == cstr)
    assert(false && "invalid integer");
  return result;
}}

using kernel_ptr_t = void(*)({kernel_fn_arg_types});

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t startX, uint32_t gridX, uint32_t gridY, uint32_t gridZ) {{
  std::unique_ptr<uint32_t[][3]> grids(new uint32_t[gridX * gridY * gridZ][3]);
  // TODO: which order would be more effective for cache locality?
  for (uint32_t z = 0; z < gridZ; ++z) {{
    for (uint32_t y = 0; y < gridY; ++y) {{
      for (uint32_t x = 0; x < gridX; ++x) {{
        grids[z * gridY * gridX + y * gridX + x][0] = startX + x;
        grids[z * gridY * gridX + y * gridX + x][1] = y;
        grids[z * gridY * gridX + y * gridX + x][2] = z;
      }}
    }}
  }}
  return grids;
}}

static void run_omp_kernels(uint32_t startX, uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads, kernel_ptr_t kernel_ptr {(', ' + arg_decls) if len(arg_decls) > 0 else ''}) {{
  // TODO: Consider using omp collapse(3) clause for simplicity?
  size_t N = gridX * gridY * gridZ;
  if (N == 1) {{
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} startX, 0, 0, gridX, gridY, gridZ);
      return;
  }}

  auto all_grids = get_all_grids(startX, gridX, gridY, gridZ);
  int omp_max_threads = 1;
  #ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
  #endif // _OPENMP
  int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

  // Don't pay OMP overhead price when a single thread is used.
  if (max_threads == 1) {{
    for (size_t i = 0; i < N; ++i) {{
      const auto [x, y, z] = all_grids[i];
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
    }}
    return;
  }}

  // For now, use the default chunk size, total iterations / max_threads.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(max_threads)
#endif // _OPENMP
  for (size_t i = 0; i < N; ++i) {{
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
  }}
}}

static inline void lower_tri_unrank(uint64_t idx, uint32_t &pidX, uint32_t &pidY) {{
  // Enumerates row-major lower-triangle tasks: for x in [1, T), y in [0, x).
  uint64_t x = static_cast<uint64_t>((1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(idx))) / 2.0);
  uint64_t base = x * (x - 1) / 2;
  while (base > idx) {{
    --x;
    base = x * (x - 1) / 2;
  }}
  while ((x + 1) * x / 2 <= idx) {{
    ++x;
    base = x * (x - 1) / 2;
  }}
  pidX = static_cast<uint32_t>(x);
  pidY = static_cast<uint32_t>(idx - base);
}}

static void run_omp_lower_tri_2d(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads, kernel_ptr_t kernel_ptr {(', ' + arg_decls) if len(arg_decls) > 0 else ''}) {{
  if (gridX != gridY || gridX == 0)
    return;
  uint64_t row_tasks = static_cast<uint64_t>(gridX - 1) * gridZ;
  if (row_tasks == 0)
    return;

  int omp_max_threads = 1;
  #ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
  #endif // _OPENMP
  int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

  if (max_threads == 1) {{
    for (uint64_t task = 0; task < row_tasks; ++task) {{
      uint32_t z = static_cast<uint32_t>(task / (gridX - 1));
      uint32_t x = static_cast<uint32_t>(task - static_cast<uint64_t>(z) * (gridX - 1)) + 1;
      for (uint32_t y = 0; y < x; ++y) {{
        (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
      }}
    }}
    return;
  }}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads)
#endif // _OPENMP
  for (uint64_t task = 0; task < row_tasks; ++task) {{
    uint32_t z = static_cast<uint32_t>(task / (gridX - 1));
    uint32_t x = static_cast<uint32_t>(task - static_cast<uint64_t>(z) * (gridX - 1)) + 1;
    for (uint32_t y = 0; y < x; ++y) {{
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
    }}
  }}
}}

static void run_omp_diagonal_2d(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads, kernel_ptr_t kernel_ptr {(', ' + arg_decls) if len(arg_decls) > 0 else ''}) {{
  uint32_t diag = std::min(gridX, gridY);
  uint64_t num_tasks = static_cast<uint64_t>(diag) * gridZ;
  if (num_tasks == 0)
    return;

  int omp_max_threads = 1;
  #ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
  #endif // _OPENMP
  int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

  if (max_threads == 1) {{
    for (uint64_t task = 0; task < num_tasks; ++task) {{
      uint32_t z = static_cast<uint32_t>(task / diag);
      uint32_t x = static_cast<uint32_t>(task - static_cast<uint64_t>(z) * diag);
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, x, z, gridX, gridY, gridZ);
    }}
    return;
  }}

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(max_threads)
#endif // _OPENMP
  for (uint64_t task = 0; task < num_tasks; ++task) {{
    uint32_t z = static_cast<uint32_t>(task / diag);
    uint32_t x = static_cast<uint32_t>(task - static_cast<uint64_t>(z) * diag);
    (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, x, z, gridX, gridY, gridZ);
  }}
}}

// Op codes for run_omp_region_2d: comparison pidY OP pidX
//   0=LT(y<x)  1=LE(y<=x)  2=EQ(y==x)  3=GE(y>=x)  4=GT(y>x)  5=NE(y!=x)
static void run_omp_region_2d(int op, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                              int num_threads, kernel_ptr_t kernel_ptr {(', ' + arg_decls) if len(arg_decls) > 0 else ''}) {{
  // Fast path: delegate to the optimised specialised runners for the two
  // most common cases (lower triangle and diagonal).
  if (op == 0) {{
    run_omp_lower_tri_2d(gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"arg{i}" for i in kernel_fn_args)) if len(kernel_fn_args) > 0 else ''});
    return;
  }}
  if (op == 2) {{
    run_omp_diagonal_2d(gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"arg{i}" for i in kernel_fn_args)) if len(kernel_fn_args) > 0 else ''});
    return;
  }}

  // Generic path: outer loop over (x, z) pairs, inner loop over y with condition.
  uint64_t outer_tasks = static_cast<uint64_t>(gridX) * gridZ;
  if (outer_tasks == 0) return;

  int omp_max_threads = 1;
  #ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
  #endif // _OPENMP
  int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

  if (max_threads == 1) {{
    for (uint64_t task = 0; task < outer_tasks; ++task) {{
      uint32_t z = static_cast<uint32_t>(task / gridX);
      uint32_t x = static_cast<uint32_t>(task - static_cast<uint64_t>(z) * gridX);
      if (op == 5) {{ // NE: [0, x) ∪ [x+1, gridY)
        for (uint32_t y = 0; y < x && y < gridY; ++y)
          (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
        for (uint32_t y = x + 1; y < gridY; ++y)
          (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
      }} else {{
        uint32_t y_lo, y_hi;
        switch (op) {{
        case 1: y_lo = 0;     y_hi = x + 1; break; // LE: [0, x]
        case 3: y_lo = x;     y_hi = gridY; break;  // GE: [x, gridY)
        case 4: y_lo = x + 1; y_hi = gridY; break;  // GT: [x+1, gridY)
        default: y_lo = 0; y_hi = 0; break;
        }}
        for (uint32_t y = y_lo; y < y_hi; ++y)
          (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
      }}
    }}
    return;
  }}

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads)
  #endif // _OPENMP
  for (uint64_t task = 0; task < outer_tasks; ++task) {{
    uint32_t z = static_cast<uint32_t>(task / gridX);
    uint32_t x = static_cast<uint32_t>(task - static_cast<uint64_t>(z) * gridX);
    if (op == 5) {{
      for (uint32_t y = 0; y < x && y < gridY; ++y)
        (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
      for (uint32_t y = x + 1; y < gridY; ++y)
        (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
    }} else {{
      uint32_t y_lo, y_hi;
      switch (op) {{
      case 1: y_lo = 0;     y_hi = x + 1; break;
      case 3: y_lo = x;     y_hi = gridY; break;
      case 4: y_lo = x + 1; y_hi = gridY; break;
      default: y_lo = 0; y_hi = 0; break;
      }}
      for (uint32_t y = y_lo; y < y_hi; ++y)
        (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
    }}
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  // Extract num_threads metadata.
  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_cpu_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};
  run_omp_kernels(0, gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature_without_constexprs.items())) if len(signature_without_constexprs) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyObject* launch_offset(PyObject* self, PyObject* args) {{
  int startX, gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{offset_format}\", &startX, &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_cpu_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};
  run_omp_kernels(startX, gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature_without_constexprs.items())) if len(signature_without_constexprs) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

static PyObject* launch_lower_tri_2d(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_cpu_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};
  run_omp_lower_tri_2d(gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature_without_constexprs.items())) if len(signature_without_constexprs) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

static PyObject* launch_diagonal_2d(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_cpu_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};
  run_omp_diagonal_2d(gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature_without_constexprs.items())) if len(signature_without_constexprs) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

static PyObject* launch_region_2d(PyObject* self, PyObject* args) {{
  int op, gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{offset_format}\", &op, &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_cpu_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};
  run_omp_region_2d(op, gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature_without_constexprs.items())) if len(signature_without_constexprs) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

static PyObject* launch_dual(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  int n_arg_idx, block_arg_idx;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *main_kernel_metadata = NULL;
  PyObject *tail_kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pMainKrnl;
  void* pTailKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"iiiOKOKOOOOii{args_format}\", &gridX, &gridY, &gridZ,
                                       &py_obj_stream, &pMainKrnl, &main_kernel_metadata,
                                       &pTailKrnl, &tail_kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook,
                                       &n_arg_idx, &block_arg_idx {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t main_kernel_ptr = reinterpret_cast<kernel_ptr_t>(pMainKrnl);
  kernel_ptr_t tail_kernel_ptr = reinterpret_cast<kernel_ptr_t>(pTailKrnl);

  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(tail_kernel_metadata, "num_cpu_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};

  auto get_int_arg = [&](int idx, int64_t &out) -> bool {{
    switch (idx) {{
{''.join([f'''    case {i}: {{
      {'if (!PyLong_Check(arg' + str(i) + ')) return false; out = PyLong_AsLongLong(arg' + str(i) + '); if (PyErr_Occurred()) { PyErr_Clear(); return false; } return true;' if ty == "constexpr" else ('return false;' if ty[0] == '*' or ty in ("fp16", "bf16", "fp32", "f32", "fp64") else 'out = static_cast<int64_t>(arg' + str(i) + '); return true;')}
    }}
''' for i, ty in signature.items()])}    default:
      return false;
    }}
  }};

  bool use_dual = false;
  int64_t n_val = 0;
  int64_t block_val = 0;
  uint32_t full_tiles = 0;
  bool has_tail = false;
  if (gridY == 1 && gridZ == 1 &&
      get_int_arg(n_arg_idx, n_val) && get_int_arg(block_arg_idx, block_val) &&
      block_val > 0 && n_val >= 0) {{
    int64_t expected_grid = (n_val + block_val - 1) / block_val;
    if (gridX == expected_grid) {{
      full_tiles = static_cast<uint32_t>(n_val / block_val);
      has_tail = n_val != static_cast<int64_t>(full_tiles) * block_val;
      use_dual = true;
    }}
  }}

  if (use_dual) {{
    uint32_t total = static_cast<uint32_t>(gridX);
    int omp_max_threads_dual = 1;
    #ifdef _OPENMP
    omp_max_threads_dual = omp_get_max_threads();
    #endif
    int max_threads_dual = (num_threads > 0) ? num_threads : omp_max_threads_dual;
    if (max_threads_dual == 1) {{
      for (uint32_t i = 0; i < total; ++i) {{
        kernel_ptr_t kptr = (i < full_tiles) ? main_kernel_ptr : tail_kernel_ptr;
        (*kptr)({(', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=='*' else f"arg{i}" for i, ty in signature_without_constexprs.items()) + ', ') if len(signature_without_constexprs) > 0 else ''}i, 0, 0, total, 1, 1);
      }}
    }} else {{
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(max_threads_dual)
      #endif
      for (uint32_t i = 0; i < total; ++i) {{
        kernel_ptr_t kptr = (i < full_tiles) ? main_kernel_ptr : tail_kernel_ptr;
        (*kptr)({(', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=='*' else f"arg{i}" for i, ty in signature_without_constexprs.items()) + ', ') if len(signature_without_constexprs) > 0 else ''}i, 0, 0, total, 1, 1);
      }}
    }}
  }} else {{
    run_omp_kernels(0, gridX, gridY, gridZ, num_threads, tail_kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature_without_constexprs.items())) if len(signature_without_constexprs) > 0 else ''});
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

// 2-D tail guard: main / m_tail / n_tail / corner
static PyObject* launch_quad(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  int m_n_arg_idx, m_block_arg_idx, n_n_arg_idx, n_block_arg_idx;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *main_kernel_metadata = NULL;
  PyObject *m_tail_kernel_metadata = NULL;
  PyObject *n_tail_kernel_metadata = NULL;
  PyObject *corner_kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pMainKrnl;
  void* pMTailKrnl;
  void* pNTailKrnl;
  void* pCornerKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"iiiOKOKOKOKOOOOiiii{args_format}\", &gridX, &gridY, &gridZ,
                                       &py_obj_stream,
                                       &pMainKrnl, &main_kernel_metadata,
                                       &pMTailKrnl, &m_tail_kernel_metadata,
                                       &pNTailKrnl, &n_tail_kernel_metadata,
                                       &pCornerKrnl, &corner_kernel_metadata,
                                       &launch_metadata, &launch_enter_hook, &launch_exit_hook,
                                       &m_n_arg_idx, &m_block_arg_idx,
                                       &n_n_arg_idx, &n_block_arg_idx
                                       {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t main_kernel_ptr   = reinterpret_cast<kernel_ptr_t>(pMainKrnl);
  kernel_ptr_t m_tail_kernel_ptr = reinterpret_cast<kernel_ptr_t>(pMTailKrnl);
  kernel_ptr_t n_tail_kernel_ptr = reinterpret_cast<kernel_ptr_t>(pNTailKrnl);
  kernel_ptr_t corner_kernel_ptr = reinterpret_cast<kernel_ptr_t>(pCornerKrnl);

  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(corner_kernel_metadata, "num_cpu_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  if (launch_enter_hook != Py_None) {{
    PyObject* hargs = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, hargs);
    Py_DECREF(hargs);
    if (!ret) return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};

  auto get_int_arg_q = [&](int idx, int64_t &out) -> bool {{
    switch (idx) {{
{''.join([f'''    case {i}: {{
      {'if (!PyLong_Check(arg' + str(i) + ')) return false; out = PyLong_AsLongLong(arg' + str(i) + '); if (PyErr_Occurred()) { PyErr_Clear(); return false; } return true;' if ty == "constexpr" else ('return false;' if ty[0] == '*' or ty in ("fp16", "bf16", "fp32", "f32", "fp64") else 'out = static_cast<int64_t>(arg' + str(i) + '); return true;')}
    }}
''' for i, ty in signature.items()])}    default:
      return false;
    }}
  }};

  bool use_quad = false;
  int64_t m_val = 0, m_block_val = 0, n_val = 0, n_block_val = 0;
  uint32_t full_m = 0, full_n = 0;
  if (get_int_arg_q(m_n_arg_idx, m_val) && get_int_arg_q(m_block_arg_idx, m_block_val) &&
      get_int_arg_q(n_n_arg_idx, n_val) && get_int_arg_q(n_block_arg_idx, n_block_val) &&
      m_block_val > 0 && n_block_val > 0 && m_val >= 0 && n_val >= 0) {{
    int64_t exp_gx = (m_val + m_block_val - 1) / m_block_val;
    int64_t exp_gy = (n_val + n_block_val - 1) / n_block_val;
    if (gridX == exp_gx && gridY == exp_gy && gridZ == 1) {{
      full_m = static_cast<uint32_t>(m_val / m_block_val);
      full_n = static_cast<uint32_t>(n_val / n_block_val);
      use_quad = true;
    }}
  }}

  if (use_quad) {{
    int omp_max_threads_q = 1;
    #ifdef _OPENMP
    omp_max_threads_q = omp_get_max_threads();
    #endif
    int max_threads_q = (num_threads > 0) ? num_threads : omp_max_threads_q;
    if (max_threads_q == 1) {{
      for (uint32_t y = 0; y < static_cast<uint32_t>(gridY); ++y) {{
        for (uint32_t x = 0; x < static_cast<uint32_t>(gridX); ++x) {{
          bool xt = (x == full_m), yt = (y == full_n);
          kernel_ptr_t kptr = (!xt && !yt) ? main_kernel_ptr
                            : ( xt && !yt) ? m_tail_kernel_ptr
                            : (!xt &&  yt) ? n_tail_kernel_ptr
                            :                corner_kernel_ptr;
          (*kptr)({(', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=='*' else f"arg{i}" for i, ty in signature_without_constexprs.items()) + ', ') if len(signature_without_constexprs) > 0 else ''}x, y, 0, gridX, gridY, 1);
        }}
      }}
    }} else {{
      uint32_t total = static_cast<uint32_t>(gridX) * static_cast<uint32_t>(gridY);
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(max_threads_q)
      #endif
      for (uint32_t i = 0; i < total; ++i) {{
        uint32_t x = i % static_cast<uint32_t>(gridX);
        uint32_t y = i / static_cast<uint32_t>(gridX);
        bool xt = (x == full_m), yt = (y == full_n);
        kernel_ptr_t kptr = (!xt && !yt) ? main_kernel_ptr
                          : ( xt && !yt) ? m_tail_kernel_ptr
                          : (!xt &&  yt) ? n_tail_kernel_ptr
                          :                corner_kernel_ptr;
        (*kptr)({(', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=='*' else f"arg{i}" for i, ty in signature_without_constexprs.items()) + ', ') if len(signature_without_constexprs) > 0 else ''}x, y, 0, gridX, gridY, 1);
      }}
    }}
  }} else {{
    run_omp_kernels(0, gridX, gridY, gridZ, num_threads, corner_kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature_without_constexprs.items())) if len(signature_without_constexprs) > 0 else ''});
  }}

  if (launch_exit_hook != Py_None) {{
    PyObject* hargs = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, hargs);
    Py_DECREF(hargs);
    if (!ret) return NULL;
  }}

  if (PyErr_Occurred()) {{ return NULL; }}
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{"launch_offset", launch_offset, METH_VARARGS, "Entry point for kernels with shifted program_id(0)"}},
  {{"launch_lower_tri_2d", launch_lower_tri_2d, METH_VARARGS, "Entry point for lower-triangle PID-region dispatch"}},
  {{"launch_diagonal_2d", launch_diagonal_2d, METH_VARARGS, "Entry point for diagonal PID-region dispatch"}},
  {{"launch_region_2d", launch_region_2d, METH_VARARGS, "Generic 2D pid-region dispatch (op-code selects tile set)"}},
  {{"launch_dual", launch_dual, METH_VARARGS, "Entry point for CPU tail-guard dual-kernel dispatch"}},
  {{"launch_quad", launch_quad, METH_VARARGS, "Entry point for CPU 2D tail-guard quad-kernel dispatch"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_cpu_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_cpu_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class CPULauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids)
        mod = compile_module_from_src(src, "__triton_cpu_launcher")
        self.launch = mod.launch
        self.launch_offset = mod.launch_offset
        self.launch_lower_tri_2d = mod.launch_lower_tri_2d
        self.launch_diagonal_2d = mod.launch_diagonal_2d
        self.launch_region_2d = mod.launch_region_2d
        self.launch_dual = mod.launch_dual
        self.launch_quad = mod.launch_quad

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.knobs.runtime.launch_enter_hook = None
        triton.knobs.runtime.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.knobs.runtime.launch_enter_hook = lambda arg: self._enter_hook()
        triton.knobs.runtime.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


class CPUDriver(DriverBase):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu", self.get_current_device())

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        # Capability and warp size are zeros for CPU.
        # TODO: GPUTarget naming isn't obviously good.
        cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        return GPUTarget("cpu", cpu_arch, 0)

    def get_device_interface(self):
        return CPUDeviceInterface()

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench

        def do_bench_cpu(*args, **kwargs):
            if not 'measure_time_with_hooks' in kwargs:
                kwargs['measure_time_with_hooks'] = True
            return do_bench(*args, **kwargs)

        return do_bench_cpu

    def get_empty_cache_for_benchmark(self):
        import torch

        # A typical LLC size for high-end server CPUs are ~400MB.
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    # TODO maybe CPU should do anything here
    def clear_cache(self, cache):
        cache.zero_()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

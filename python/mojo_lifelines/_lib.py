from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_LIFELINES_LIB", os.path.join(ROOT, "dist", "libmojo-lifelines.so"))

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "ml_event_table": ([I] * 16, I),
    "ml_weibull_fit": ([I, I, I, I, I, I, F], I),
    "ml_cox_efron": ([I] * 12, F),
    "ml_predict_log_hazard": ([I, I, I, I, I], None),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError(f"Mojo library not found at {LIB}; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(values, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(values)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise ValueError("complex values cannot be represented as float64")
    if np.issubdtype(source.dtype, np.integer):
        limit = 1 << 53
        if np.any(source > limit) or np.any(source < -limit):
            raise ValueError("integer values outside the exact float64 range are not supported")
    if copy:
        return np.array(values, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(values, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("native arguments must be NumPy arrays")
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise TypeError("native arrays must be C-contiguous float64")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("native arrays must have a non-null data pointer")
    return address

from __future__ import annotations

import numpy as np


INT32_MAX = int(np.iinfo(np.int32).max)


def index_dtype_for_shape(shape):
    """Return the smallest signed integer dtype able to hold flat cell indices."""
    rows, cols = int(shape[0]), int(shape[1])
    max_index = rows * cols - 1
    return np.int32 if max_index <= INT32_MAX else np.int64


def index_dtype_for_size(size: int):
    return np.int32 if int(size) - 1 <= INT32_MAX else np.int64


def dtype_name(dtype) -> str:
    return np.dtype(dtype).name

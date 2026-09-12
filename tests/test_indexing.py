import numpy as np

from pyslopeunits.indexing import index_dtype_for_shape


def test_index_dtype_switches_above_int32_flat_index_limit():
    assert index_dtype_for_shape((1000, 1000)) == np.int32
    assert index_dtype_for_shape((50000, 50000)) == np.int64

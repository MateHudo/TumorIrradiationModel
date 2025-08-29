import numpy as np

def upscale_matrix(matrix, upscale):
    """
    Upscale a matrix by repeating each element into an (upscale x upscale) block.
    If input is a scalar, returns a (upscale x upscale) matrix filled with that value.
    """
    arr = np.array(matrix)
    if arr.ndim == 0:
        # Scalar input
        return np.full((upscale, upscale), arr)
    else:
        # Matrix input
        return np.kron(arr, np.ones((upscale, upscale), dtype=arr.dtype))
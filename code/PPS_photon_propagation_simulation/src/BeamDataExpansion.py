import numpy as np

def expand_to_4Nbeams(matrix_3d):
    """
    Expand a 3D matrix of shape ((N+1)//2, N, N) to (4N, N, N) using the described flipping and transposing logic.

    Args:
        matrix_3d (np.ndarray): Input 3D matrix of shape ((N+1)//2, N, N), N odd.

    Returns:
        np.ndarray: Expanded 3D matrix of shape (4N, N, N).
    """
    n_half = matrix_3d.shape[0]
    N = matrix_3d.shape[1]
    assert N % 2 == 1, "N must be odd"
    assert matrix_3d.shape == (n_half, N, N)

    # Step 1: Expand to (N, N, N)
    expanded = np.zeros((N, N, N), dtype=matrix_3d.dtype)
    for k in range(n_half):
        expanded[k] = matrix_3d[k]
    for k in range((N-1)//2):
        flipped = matrix_3d[k][::-1, :]
        expanded[N - k - 1] = flipped

    # Step 2: Expand to (4N, N, N)
    Nbeams = 4 * N
    SIF_total = np.zeros((Nbeams, N, N), dtype=matrix_3d.dtype)
    SIF_total[:N] = expanded
    for k in range(N):
        mat = SIF_total[k]
        # For SIF_total[N + k], flip columns (reverse each row)
        SIF_total[N + k] = mat[:, ::-1]
        # For SIF_total[2*N + k], transpose and flip columns
        SIF_total[3*N - k-1] = mat.T[:, ::-1]
        # For SIF_total[3*N + k], transpose and flip rows
        SIF_total[3*N + k] = mat.T[::-1, :]
    return SIF_total


def repair_wrong_expansion(matrix):
    num_beams = matrix.shape[0] # number of beams in one direction
    num_side = num_beams // 4
    #num_beams = 4 * N # four direction we have
    #n_upper = (N+1)//2

    repaired = matrix
    # we need to switch order of factos/matrices for 3rd line of beams, so we need to flip the matrices
    for k in range(num_side):
        repaired[2*num_side + k] = matrix[k].T[:, ::-1]
    return repaired

def repair_SIF_matrix(filepath):
    matrix = np.load(filepath)
    repaired = repair_wrong_expansion(matrix)
    return repaired

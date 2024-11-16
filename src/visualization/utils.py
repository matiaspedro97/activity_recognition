import math
import numpy as np

def subplot_dim_optm(dim: int):
    # most squared matrix
    n_dim, m_dim = int(np.sqrt(dim)), int(np.sqrt(dim))

    # refactor n dim
    n_dim += math.ceil((dim - m_dim ** 2) / n_dim)

    return n_dim, m_dim
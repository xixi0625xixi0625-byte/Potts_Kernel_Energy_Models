import numpy as np
from itertools import product

def get_all_sequences(L, q):

    all_sequences = np.array(list(product(range(q), repeat=L)))
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]

    return all_sequences
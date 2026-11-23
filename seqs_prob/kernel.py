import numpy as np
from seqs_prob.hamming_distance import hamming_distance

#Kernel K(x, u)=sigma_E^2 * exp(-H(x, u) / C)
#return the kernel matrix K of shape (len(seqs1), len(seqs2))

def kernel(seqs1, seqs2, sigma_E_squared, c):
    H = hamming_distance(seqs1, seqs2)
    K = sigma_E_squared * np.exp(-H / c)
    return K
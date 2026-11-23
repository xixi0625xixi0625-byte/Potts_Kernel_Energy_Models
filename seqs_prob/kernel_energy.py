import numpy as np
from seqs_prob.kernel import kernel

#E(x) = sum_u K(x, u) alpha_u
def kernel_energy(seqs, alpha, U, sigma_E_squared, c):
    return kernel(seqs, U, sigma_E_squared, c) @ alpha
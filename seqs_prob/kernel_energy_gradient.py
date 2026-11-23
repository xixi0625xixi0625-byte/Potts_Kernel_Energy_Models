import numpy as np
from seqs_prob.kernel import kernel

#dE(x)/alpha_u = K(x, u)
def kernel_energy_gradient(seqs, U, sigma_E_squared, c):
    return kernel(seqs, U, sigma_E_squared, c)
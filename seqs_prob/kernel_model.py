import numpy as np
from seqs_prob.hamming_distance import hamming_distance

#Kernel K(x, u)=sigma_E^2 * exp(-H(x, u) / C)
#return the kernel matrix K of shape (len(seqs1), len(seqs2))

def kernel(seqs1, seqs2, sigma_E_squared, c):
    H = hamming_distance(seqs1, seqs2)
    K = sigma_E_squared * np.exp(-H / c)
    return K

#E(x) = sum_u K(x, u) alpha_u
def kernel_energy(seqs, alpha, U, sigma_E_squared, c):
    return kernel(seqs, U, sigma_E_squared, c) @ alpha


#dE(x)/alpha_u = K(x, u)
def kernel_energy_gradient(seqs, U, sigma_E_squared, c):
    return kernel(seqs, U, sigma_E_squared, c)


#alpha ~ N(0, K_uu^-1)
#log_P(alpha) = -1/2 alpha^T K_uu alpha
def kernel_prior(alpha, K_uu):
    return -0.5 * alpha @ K_uu @ alpha


#d/dalpha [-1/2 alpha^T K_uu alpha] = -K_uu alpha
def kernel_prior_gradient(alpha, K_uu):
    return -K_uu @ alpha


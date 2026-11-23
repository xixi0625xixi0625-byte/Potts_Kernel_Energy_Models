import numpy as np

#d/dalpha [-1/2 alpha^T K_uu alpha] = -K_uu alpha
def kernel_prior_gradient(alpha, K_uu):
    return -K_uu @ alpha
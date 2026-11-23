import numpy as np

#alpha ~ N(0, K_uu^-1)
#log_P(alpha) = -1/2 alpha^T K_uu alpha
def kernel_prior(alpha, K_uu):
    return -0.5 * alpha @ K_uu @ alpha

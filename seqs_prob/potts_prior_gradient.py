import numpy as np

#d/dh = -h / sigma_h^2
#d/dJ = -J / sigma_J^2
def potts_prior_gradient(theta, L, q, sigma_h_squared, sigma_J_squared):
    h, J = theta[:L*q], theta[L*q:]
    return np.concatenate([-h/sigma_h_squared, -J/sigma_J_squared])
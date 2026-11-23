import numpy as np

#prior
#log P(h) + log P(J) = -sum h^2 / (2 sigma_h^2) - sum J^2 / (2 sigma_J^2)
def potts_prior(theta, L, q, sigma_h_squared, sigma_J_squared):
    h, J = theta[:L*q], theta[L*q:]
    return -0.5 * np.sum(h**2)/sigma_h_squared - 0.5 * np.sum(J**2)/sigma_J_squared

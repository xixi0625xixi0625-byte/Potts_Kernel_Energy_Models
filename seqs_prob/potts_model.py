import numpy as np
from scipy.special import logsumexp

#E(a|x_i) = h_i(a) + sum_{j != i} J_{ij}(a, x_j)
def potts_conditional_energy(seqs, theta, L, q):
    h = theta[:L*q].reshape((L, q))
    J = theta[L*q:].reshape((L, L, q, q))
    E_conditional = np.zeros((len(seqs), L, q))
    for i in range(L):
        #h_i(a) for every allel a 
        E_conditional[:, i, :] += h[i]
        #add J_ij(a, x_j) for every position j != i
        for j in range(L):
            if j != i:
                if i < j:
                    J_ij = J[i, j]
                else:
                    J_ij = J[j, i].T
                E_conditional[:, i, :] += J_ij[:, seqs[:, j]].T
                print(E_conditional)
    return E_conditional


#dE/dh_i(a) =1 if x_i = a, 0 otherwise
#dE/dJ_ij(a,b) = 1 if x_i = a and x_j = b, 0 otherwise

def potts_energy_gradient(seqs, L, q):
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    M = len(seqs)
    rows = np.arange(M)
    gradient_h = np.zeros((M, L, q))
    gradient_J = np.zeros((M, L, L, q, q))
    for i in range(L):
        gradient_h[rows, i, seqs[:, i]] = 1
    for i, j in pairs:
        gradient_J[rows, i, j, seqs[:, i], seqs[:, j]] = 1
    return np.concatenate([gradient_h.reshape(M, -1), gradient_J.reshape(M, -1)], axis=1)


def potts_energy(seqs, theta, L, q):
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    h = theta[:L*q].reshape(L,q)
    J = theta[L*q:].reshape(L,L,q,q)
    E = h[np.arange(L), seqs].sum(axis=1)
    for i, j in pairs:
        E += J[i, j, seqs[:, i], seqs[:, j]]
    return E



#log PL = sum_n sum_i [E(x_i | x_-i) - log Z_i]
#log Z_i = log sum_a exp(E(a | x_-i))
def potts_log_pseudolikelihood(theta, seqs, L, q):
    E_conditional = potts_conditional_energy(seqs, theta, L, q)
    log_Z = logsumexp(E_conditional, axis=2)
    #E(x_i | x_-i): only the letter that is really at position i
    E_true = np.zeros((len(seqs), L))
    for n in range(len(seqs)):
        for i in range(L):
            E_true[n, i] = E_conditional[n, i, seqs[n, i]]
    return np.sum(E_true) - np.sum(log_Z)


#d/dh = -h / sigma_h^2
#d/dJ = -J / sigma_J^2
def potts_prior_gradient(theta, L, q, sigma_h_squared, sigma_J_squared):
    h, J = theta[:L*q], theta[L*q:]
    return np.concatenate([-h/sigma_h_squared, -J/sigma_J_squared])



#prior
#log P(h) + log P(J) = -sum h^2 / (2 sigma_h^2) - sum J^2 / (2 sigma_J^2)
def potts_prior(theta, L, q, sigma_h_squared, sigma_J_squared):
    h, J = theta[:L*q], theta[L*q:]
    return -0.5 * np.sum(h**2)/sigma_h_squared - 0.5 * np.sum(J**2)/sigma_J_squared
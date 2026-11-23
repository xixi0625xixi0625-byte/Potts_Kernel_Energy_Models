import numpy as np
from scipy.special import logsumexp
from seqs_prob.potts_conditional_energy import potts_conditional_energy


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
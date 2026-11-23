import numpy as np
from scipy.special import logsumexp

#log p = sum_n E(x_n) - N log Z
#energy(seqs, theta): the energy of every sequence in seqs
def log_likelihood(theta, energy, encoded_sequences, all_sequences):
    N = len(encoded_sequences)
    log_Z = logsumexp(energy(all_sequences, theta))
    return np.sum(energy(encoded_sequences, theta)) - N * log_Z
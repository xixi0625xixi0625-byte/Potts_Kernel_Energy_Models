import numpy as np
from scipy.special import logsumexp, softmax

#log p = sum_n E(x_n) - N log Z
#energy(seqs, theta): the energy of every sequence in seqs
def log_likelihood(theta, energy, encoded_sequences, all_sequences):
    N = len(encoded_sequences)
    log_Z = logsumexp(energy(all_sequences, theta))
    return np.sum(energy(encoded_sequences, theta)) - N * log_Z


#gradient = sum_n dE(x_n) - N sum_x p(x) dE(x)
#energy(seqs, theta): energy of every sequence in seqs
#energy_gradient(seqs): dE/dtheta, one row for each sequence
def log_likelihood_gradient(theta, energy, energy_gradient, encoded_sequences, all_sequences):
    N = len(encoded_sequences)
    P = softmax(energy(all_sequences, theta))   #p(x) = exp(E(x)) / Z
    return energy_gradient(encoded_sequences).sum(axis=0) - N * P @ energy_gradient(all_sequences)
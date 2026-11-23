import numpy as np
from scipy.special import softmax


#gradient = sum_n dE(x_n) - N sum_x p(x) dE(x)
#energy(seqs, theta): energy of every sequence in seqs
#energy_gradient(seqs): dE/dtheta, one row for each sequence
def log_likelihood_gradient(theta, energy, energy_gradient, encoded_sequences, all_sequences):
    N = len(encoded_sequences)
    P = softmax(energy(all_sequences, theta))   #p(x) = exp(E(x)) / Z
    return energy_gradient(encoded_sequences).sum(axis=0) - N * P @ energy_gradient(all_sequences)
import numpy as np
from scipy.special import logsumexp
from seqs_prob.single_site_variants import single_site_variants


#log PL = sum_n sum_p [E(true letter in the sequence) - log sum_a exp(E(x_p = a))]
#uses the energy of the whole sequence, so it works for potts and kernel
def log_pseudolikelihood(theta, energy, encoded_sequences, q):
    total = 0
    for x in encoded_sequences:
        for p in range(len(x)):
            E = energy(single_site_variants(x, p, q), theta)   #E including every letter at position p
            total += E[x[p]] - logsumexp(E)
    return total
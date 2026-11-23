import numpy as np
from scipy.special import softmax
from seqs_prob.single_site_variants import single_site_variants


#gradient = sum_n sum_p [dE(true letter) - sum_a P(a | x_-p) dE(x_p = a)]
#P(a | x_-p) = exp(E(x_p = a)) / sum_a exp(E(x_p = a))
def log_pseudolikelihood_gradient(theta, energy, energy_gradient, encoded_sequences, q):
    gradient = np.zeros(len(theta))
    for x in encoded_sequences:
        for p in range(len(x)):
            variants = single_site_variants(x, p, q)
            E = energy(variants, theta)
            dE = energy_gradient(variants)
            P = softmax(E)
            #true letter
            gradient += dE[x[p]]
            #minus the estimation of the model
            for a in range(q):
                gradient -= P[a] * dE[a]
    return gradient
import numpy as np

#sequence x: on position p: change to every allel, other positions fixed
#getting q new sequences, which is used to compare the energy when putting different allels on position p
#returns (q, L)
#row a is x with x_p = a, everything else stays the same
def single_site_variants(x, p, q):
    variants = np.array([x] * q)
    for a in range(q):
        variants[a, p] = a
    return variants

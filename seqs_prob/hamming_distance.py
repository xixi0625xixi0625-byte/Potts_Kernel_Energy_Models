import numpy as np

def hamming_distance(seqs1, seqs2):
    return np.sum(seqs1[:, None, :] != seqs2[None, :, :], axis=2)
#returns matrix of hamming distances (len(seqs1), len(seqs2))
#number of positions where two sequences are different
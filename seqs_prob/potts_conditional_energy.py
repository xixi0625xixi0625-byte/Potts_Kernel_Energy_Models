import numpy as np

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
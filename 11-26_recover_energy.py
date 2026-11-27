import numpy as np
from scipy.special import softmax
from itertools import product
from scipy.optimize import minimize
from scipy.special import logsumexp

a = "ACGT"
q = len(a)
L = 6

all_sequences = np.array(list(product(range(q), repeat=L)))
pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]

sigma_h_squared = 1
sigma_J_squared = 1
np.random.seed(0)

# h
h = np.random.normal(0, sigma_h_squared, size=(L, q))
# J
J = np.zeros((L, L, q, q))
for i, j in pairs:
    J[i, j] = np.random.normal(0, sigma_J_squared, size=(q, q))

theta = np.concatenate([h.flatten(), J.flatten()])

#E(X) = sum_i h_i(x_i) + sum_(i<j) J_{ij}(x_i, x_j)
def potts_energy(seqs, theta):
    h = theta[:L*q].reshape(L,q)
    J = theta[L*q:].reshape(L,L,q,q)
    E = h[np.arange(L), seqs].sum(axis=1)
    for i, j in pairs:
        E += J[i, j, seqs[:, i], seqs[:, j]]
    return E

# Energy for every possible sequences
E_all = potts_energy(all_sequences, theta)
P = softmax(E_all)

folder = "C:/Users/Lenovo/Desktop/CSHL/VScode/"
np.save(folder + "theta_true.npy", theta)
np.save(folder + "energy_true.npy", E_all)
np.savetxt(folder + "theta_true.txt", theta)
np.savetxt(folder + "energy_true.txt", E_all)

index = np.random.choice(len(all_sequences), size=20000, p=P)
sample = all_sequences[index]

encoded_sequences = sample
N, L = encoded_sequences.shape

#unique sequences in the data and how many times each one shows up
#loop over U instead of all 20000 sequences, then multiply by counts
U, counts = np.unique(encoded_sequences, axis=0, return_counts=True)

#E(X)
#option1 - potts

#theta = {h_i, J_ij}
n_potts = L * q + L * L * q * q

#gradient for potts_energy E -> likelihood
#dE/dh_i(k) = 1 if x_i = k, 0 otherwise
#dE/dJ_{ij}(k, l) = 1 if x_i =k and x_j = l, 0 otherwise
def potts_energy_gradient(seqs):
    M = len(seqs)
    rows = np.arange(M)
    gradient_h = np.zeros((M, L, q))
    gradient_J = np.zeros((M, L, L, q, q))
    for i in range(L):
        gradient_h[rows, i, seqs[:, i]] = 1
    for i, j in pairs:
        gradient_J[rows, i, j, seqs[:, i], seqs[:, j]] = 1
    return np.concatenate([gradient_h.reshape(M, -1), gradient_J.reshape(M, -1)], axis=1)

#prior
#log P(h) + log P(J) = -sum h^2 / (2 sigma_h^2) - sum J^2 / (2 sigma_J^2)
def potts_prior(theta, sigma_h_squared, sigma_J_squared):
    h, J = theta[:L*q], theta[L*q:]
    return -0.5 * np.sum(h**2)/sigma_h_squared - 0.5 * np.sum(J**2)/sigma_J_squared

#d/dh = -h / sigma_h^2
#d/dJ = -J / sigma_J^2
def potts_prior_gradient(theta, sigma_h_squared, sigma_J_squared):
    h, J = theta[:L*q], theta[L*q:]
    return np.concatenate([-h/sigma_h_squared, -J/sigma_J_squared])

#option 2 - kernel method

#K(x, u) = sigma_E^2 * exp(-H(x, u) / c)
#H: hamming distance between sequences
#sigma_e^2 and c are fixed right now, but will be updated later
c = 1
sigma_E_squared = 1

def K(seqs1, seqs2):
    H = np.sum(seqs1[:, None, :] != seqs2[None, :, :], axis=2)
    return sigma_E_squared * np.exp(-H / c)

#inducing sequences u (the unique data sequences)
K_uu = K(U, U)

#E(X) = sum_n K(X, U) alpha_u
#theta = alpha
n_kernel = len(U)
def kernel_energy(seqs, alpha):
    return K(seqs, U) @ alpha

#gradient for kernel energy E -> likelihood
#dE/dalpha_u = K(X, U)
def kernel_energy_gradient(seqs):
    return K(seqs, U)

#prior
#alpha = N(0, K_uu -1)
#log p(alpha) = -1/2 alpha transpose K_uu alpha
def kernel_prior(alpha, K_uu):
    return -0.5 * alpha.T @ K_uu @ alpha

#gradient for the kernel prior
def kernel_prior_gradient(alpha):
    return -K_uu @ alpha

#likelihood
#option1 - likelihood

#log p = sum_n E(x_n) - N log Z
#sum_n E(x_n) = sum_u count(u) E(u)
def log_likelihood(theta, energy, U, counts, all_sequences, N):
    log_Z = logsumexp(energy(all_sequences, theta))
    return counts @ energy(U, theta) - N * log_Z

#gradient = sum_n dE(x_n) - N sum_x p(x) dE(x)
#data_gradient and all_gradient do not depend on theta, so they are computed once in fit
def log_likelihood_gradient(theta, energy, data_gradient, all_gradient, all_sequences, N):
    P = softmax(energy(all_sequences, theta))   #p(x) = exp(E(x)) / Z
    return data_gradient - N * P @ all_gradient

#option 2 - pseudolikelihood

#pseudolikelihood for potts model

#conditional probability of x_i given x_-i
#E(X_P|X_-P) = h_i(x_i) + sum_(j != i) J_ij(x_i, x_j)
#J is only stored for i < j, so for j < i use J[j, i] with the letters swapped
def potts_conditional_energy(seqs, theta):
    h = theta[:L*q].reshape(L,q)
    J = theta[L*q:].reshape(L,L,q,q)
    E_conditional_probability = np.zeros((len(seqs), L, q))
    for i in range(L):
        #h(i) for every letter a
        E_conditional_probability[:, i, :] = h[i]
        #adding J_ij(a, x_j) for every other position j
        for j in range(L):
            if j > i:
                E_conditional_probability[:, i, :] += J[i, j, :, seqs[:, j]]
            elif j < i:
                E_conditional_probability[:, i, :] += J[j, i, seqs[:, j], :]
    return E_conditional_probability

#log pseudolikelihood = sum_n sum_i log p(x_i | x_-i)
#                     = sum_n sum_i [E(x_i | x_-i) - log Z_i]
#log Z_i = log sum_a exp(E(a | x_-i))
def potts_log_pseudolikelihood(theta, U, counts):
    E_conditional_probability = potts_conditional_energy(U, theta)
    log_Z = logsumexp(E_conditional_probability, axis=2)
    #E(x_i | x_-i): only the letter that is at position i
    E_true = np.take_along_axis(E_conditional_probability, U[:, :, None], axis=2)[:, :, 0]
    return np.sum(counts[:, None] * (E_true - log_Z))

#single site variants of one sequence x at position p (like 00, 10 in epistasis)
#only position p is changed, everything else stays the same
#  variants[a] = x with x_p = a,  for every letter a in the alphabet
def single_site_variants(x, p):
    variants = np.array([x] * q)
    for a in range(q):
        variants[a, p] = a
    return variants

#pseudolikelihood for kernel model
#use the energy of the whole variant F_a
#log p(x_p | x_-p) = F_true - log sum_a exp(F_a)
def kernel_log_pseudolikelihood(theta, U, counts):
    total = 0
    for x, w in zip(U, counts):
        for p in range(L):
            F = kernel_energy(single_site_variants(x, p), theta)   #F_a for every letter a
            total += w * (F[x[p]] - logsumexp(F))
    return total

#gradient for pseudolikelihood (potts & kernel)
#  d/dtheta [E(x_p | x_-p) - log Z_p] = dF_true - sum_a P(a | x_-p) dF_a
#  P(a | x_-p) = exp(F_a) / sum_a exp(F_a)
def log_pseudolikelihood_gradient(theta, energy, energy_gradient, U, counts):
    gradient = np.zeros_like(theta)
    for x, w in zip(U, counts):
        for p in range(L):
            variants = single_site_variants(x, p)
            dF = energy_gradient(variants)
            F = dF @ theta
            P = softmax(F)
            #first term: the true letter, F_true is x itself
            #second term: sum_a P(a | x_-p) dF_a = P @ dF
            gradient += w * (dF[x[p]] - P @ dF)
    return gradient

#ML: maximize log likelihood
#BI: maximize log likelihood + log prior
def fit(energy_type, likelihood_type, inference):
    if energy_type == "potts":
        energy = potts_energy
        energy_gradient = potts_energy_gradient
        n_parameters = n_potts
    else:
        energy = kernel_energy
        energy_gradient = kernel_energy_gradient
        n_parameters = n_kernel

    #these do not change with theta, compute once
    if likelihood_type == "likelihood":
        data_gradient = counts @ energy_gradient(U)
        all_gradient = energy_gradient(all_sequences)

    #minimize -(log likelihood + log prior)
    def f(theta):
        if likelihood_type == "likelihood":
            value = log_likelihood(theta, energy, U, counts, all_sequences, N)
        elif energy_type == "potts":
            value = potts_log_pseudolikelihood(theta, U, counts)
        else:
            value = kernel_log_pseudolikelihood(theta, U, counts)

        if inference == "BI" and energy_type == "potts":
            value += potts_prior(theta, sigma_h_squared, sigma_J_squared)
        if inference == "BI" and energy_type == "kernel":
            value += kernel_prior(theta, K_uu)
        return -value

    def df(theta):
        if likelihood_type == "likelihood":
            g = log_likelihood_gradient(theta, energy, data_gradient, all_gradient, all_sequences, N)
        else:
            g = log_pseudolikelihood_gradient(theta, energy, energy_gradient, U, counts)

        if inference == "BI" and energy_type == "potts":
            g += potts_prior_gradient(theta, sigma_h_squared, sigma_J_squared)
        if inference == "BI" and energy_type == "kernel":
            g += kernel_prior_gradient(theta)
        return -g

    result = minimize(f, np.zeros(n_parameters), jac=df, method='L-BFGS-B')
    return result.x, energy

#strength_ij = sqrt(sum_(k,l) J_ij(k,l)^2), potts only
def coupling_strength(theta):
    J = theta[L*q:].reshape(L, L, q, q)
    strength = np.sqrt(np.sum(J**2, axis=(2, 3)))
    return strength + strength.T

#epistasis between position i and j on a background sequence x
#change x_i to a2 and x_j to b2
#epsilon = (E11 - E01) - (E10 - E00)
#potts: epsilon does not depend on the background
def epistasis(energy, theta, background, i, j, a2, b2):
    seqs = np.array([background, background, background, background])
    seqs[1, i] = a2
    seqs[2, j] = b2
    seqs[3, i] = a2
    seqs[3, j] = b2
    E00, E10, E01, E11 = energy(seqs, theta)
    return (E11 - E01) - (E10 - E00)

#epsilon
#over all unique data sequences as background
#and all changes at position i and j
def epistasis_matrix(energy, theta):
    strength = np.zeros((L, L))
    for i, j in pairs:
        values = []
        for x in U:
            for a2 in range(q):
                for b2 in range(q):
                    if a2 != x[i] and b2 != x[j]:
                        values.append(abs(epistasis(energy, theta, x, i, j, a2, b2)))
        strength[i, j] = np.mean(values)
        strength[j, i] = strength[i, j]
    return strength


theta_fit, energy = fit("kernel", "likelihood", "BI")
energy_fit = energy(all_sequences, theta_fit)

np.save(folder + "theta_fit.npy", theta_fit)
np.save(folder + "energy_fit.npy", energy_fit)

import matplotlib.pyplot as plt

plt.scatter(E_all, energy_fit, s = 5)
plt.xlabel("True Energy")
plt.ylabel("Fitted Energy")
plt.show()




# #measure the accuracy of each method by the KL divergence from 
# #the true distribution to the fitted one
# #compute exactly by enumerating all sequences 
# #a lower KL means the fitted model is closer to the true distribution

# #KL(P_true || P_fit) = sum_x P_true(x) * (log P_true(x) - log P_fit(x))
# logP_true = E_all - logsumexp(E_all)

# def KL_divergence(energy, theta_fit):
#     E_fit = energy(all_sequences, theta_fit)
#     logP_fit = E_fit - logsumexp(E_fit)
#     return np.sum(np.exp(logP_true) * (logP_true - logP_fit))

# combos = [("potts", "likelihood", "ML"), ("potts", "likelihood", "BI"),
#           ("potts", "pseudo", "ML"), ("potts", "pseudo", "BI"),
#           ("kernel", "likelihood", "ML"), ("kernel", "likelihood", "BI"),
#           ("kernel", "pseudo", "ML"), ("kernel", "pseudo", "BI")]

# KL = {}
# for combo in combos:
#     theta_fit, energy = fit(combo[0], combo[1], combo[2])
#     KL[combo] = KL_divergence(energy, theta_fit)
#     print(combo, KL[combo])

# #compare two combos that only differ in one thing
# pairs_to_compare = [
#     #likelihood vs pseudo
#     (("potts", "likelihood", "ML"), ("potts", "pseudo", "ML")),
#     (("potts", "likelihood", "BI"), ("potts", "pseudo", "BI")),
#     (("kernel", "likelihood", "ML"), ("kernel", "pseudo", "ML")),
#     (("kernel", "likelihood", "BI"), ("kernel", "pseudo", "BI")),
#     #ML vs BI
#     (("potts", "likelihood", "ML"), ("potts", "likelihood", "BI")),
#     (("potts", "pseudo", "ML"), ("potts", "pseudo", "BI")),
#     (("kernel", "likelihood", "ML"), ("kernel", "likelihood", "BI")),
#     (("kernel", "pseudo", "ML"), ("kernel", "pseudo", "BI")),
#     #potts vs kernel
#     (("potts", "likelihood", "ML"), ("kernel", "likelihood", "ML")),
#     (("potts", "likelihood", "BI"), ("kernel", "likelihood", "BI")),
#     (("potts", "pseudo", "ML"), ("kernel", "pseudo", "ML")),
#     (("potts", "pseudo", "BI"), ("kernel", "pseudo", "BI")),
# ]

import unittest
import numpy as np
from seqs_prob.kernel_model import kernel, kernel_energy, kernel_energy_gradient, kernel_prior_gradient

class test_kernel(unittest.TestCase):

    def test_kernel(self):

        #input 
        seqs = np.array([[0, 1, 2, 3],
                        [0, 1, 2, 0],
                        [3, 3, 3, 3]])
        sigma_E_squared = 2
        c =2

        #run the function
        K = kernel(seqs, seqs, sigma_E_squared, c)

        #expected output
        #k = 2*exp(-H/2)
        #hamming distance:[0, 1, 3],[1, 0, 4],[3, 4, 0]
        expected = np.array([[2*np.exp(-0/2), 2*np.exp(-1/2), 2*np.exp(-3/2)],
                            [2*np.exp(-1/2), 2*np.exp(-0/2), 2*np.exp(-4/2)],
                            [2*np.exp(-3/2), 2*np.exp(-4/2), 2*np.exp(-0/2)]])

        #compare the output wth the expected output
        assert np.all(K == expected)


    def test_kernel_energy(self):

        #input
        seqs = np.array([[0, 0], [0, 1]])
        U = np.array([[0, 0], [1, 1]])
        alpha = np.array([1, 2])
        sigma_E_squared =1
        c = 1

        #run the function
        energy = kernel_energy(seqs, alpha, U, sigma_E_squared, c)

        #expected output
        #E(x) = K(x, u_1)*alpha1 + K(x, u_2)*alpha2
        #k = exp(-H)
        #[0, 0]: e^0*1+e^-2*2=1+2e^-2
        #[0, 1]: e^-1*1+e^-1*2=3e^-1

        expected = np.array([1+2*np.exp(-2), 3*np.exp(-1)])

        #compare the output with the expected output
        assert np.all(energy == expected)


    def test_kernel_energy_gradient(self):
    
        #input
        seqs = np.array([[0, 0],
                            [0, 1]])
        U = np.array([[0, 0],
                        [1, 1]])
        sigma_E_squared = 1
        c = 1

        #run the function
        gradient = kernel_energy_gradient(seqs, U, sigma_E_squared, c)

        #expected output
        #dE(x)/d alpha_u = K(x, u) = exp(-H)
        #[0, 0]: distance between [0, 0] and u1[0, 0]: 0
        #[0, 0]: distance between [0, 0] and u2[1, 1]: 2
        #[0, 1]: distance between [0, 1] and u1[0, 0]: 1
        #[0, 1]: distance between [0, 1] and u2[1, 1]: 1
        expected = np.array([[np.exp(0), np.exp(-2)],
                                [np.exp(-1), np.exp(-1)]])

        #compare the output with the expected output
        assert np.all(gradient == expected)


    def test_kernel_prior_gradient(self):
    
        #input: 
        alpha = np.array([1, 2])
        K_uu = np.array([[2, 1],
                            [1, 3]])

        #run the function
        gradient = kernel_prior_gradient(alpha, K_uu)

        #expected output: 
        #-K_uu  @ alpha
        #first row:  -(2 * 1 + 1 * 2) = -4
        #second row: -(1 * 1 + 3 * 2) = -7
        expected = np.array([-4, -7])

        #compare the output with the expected output
        assert np.all(expected == gradient)

if __name__ == '__main__':
    unittest.main()
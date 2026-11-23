import unittest
import numpy as np
from seqs_prob.kernel_energy_gradient import kernel_energy_gradient

class test_kernel_energy_gradient(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
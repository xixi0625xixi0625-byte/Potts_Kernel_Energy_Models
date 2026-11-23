import unittest
import numpy as np
from seqs_prob.potts_prior import potts_prior

class test_potts_prior(unittest.TestCase):

    def test_potts_prior(self):

        #input
        L = 2
        q = 2
        h = np.array([[1, 2], 
                      [3, 4]])
        J = np.zeros((L, L, q, q))
        J[0, 1] = np.array([[1, 0],
                            [0, 2]])
        theta = np.concatenate([h.flatten(), J.flatten()])
        sigma_h_squared = 2
        sigma_J_squared = 1

        #run the function
        prior = potts_prior(theta, L, q, sigma_h_squared, sigma_J_squared)

        #expected output
        #-sum h^2 / (2 sigma_h^2) - sum J^2 / (2 sigma_J^2)
        #sum h^2 = 1^2 + 2^2 + 3^2 + 4^2 = 30     -30 / (2*2) = -7.5
        #sum J^2 = 1^2 + 2^2 = 5        -5 / (2*1) = -2.5
        #-7.5 - 2.5 = -10
        expected = -10

        #compare the output with the expected output
        assert np.all(prior == expected)

if __name__ == '__main__':
    unittest.main()
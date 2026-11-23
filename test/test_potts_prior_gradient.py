import unittest
import numpy as np
from seqs_prob.potts_prior_gradient import potts_prior_gradient

class test_potts_prior_gradient(unittest.TestCase):

    def test_potts_prior_gradient(self):

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
        gradient = potts_prior_gradient(theta, L, q, sigma_h_squared, sigma_J_squared)

        #expected output: -theta / sigma^2
        #d/dh = -h / sigma_h^2 = -[1, 2, 3, 4] / 2 = [-0.5, -1, -1.5, -2]
        #d/dJ = -J / sigma_J^2, J contains 16 numbers, only J[0,1] is not 0
        expected_h = np.array([-0.5, -1, -1.5, -2])
        expected_J = np.array([0, 0, 0, 0,       #J[0,0]
                               -1, 0, 0, -2,     #J[0,1]
                               0, 0, 0, 0,       #J[1,0]
                               0, 0, 0, 0])      #J[1,1]
        expected = np.concatenate([expected_h, expected_J])

        #compare the output with the expected output
        assert np.all(gradient == expected)


if __name__ == '__main__':
    unittest.main()
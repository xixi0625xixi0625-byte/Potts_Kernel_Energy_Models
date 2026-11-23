import unittest
import numpy as np
from seqs_prob.kernel_prior_gradient import kernel_prior_gradient

class test_kernel_prior_gradient(unittest.TestCase):

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
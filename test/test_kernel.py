import unittest
import numpy as np
from seqs_prob.kernel import kernel

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

if __name__ == '__main__':
    unittest.main()
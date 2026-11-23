import unittest
import numpy as np
from seqs_prob.potts_log_pseudolikelihood import potts_log_pseudolikelihood

class test_potts_log_pseudolikelihood(unittest.TestCase):

    def test_potts_log_pseudolikelihood(self):

        #input
        seqs = np.array([[0, 1]])
        L =2
        q = 2
        theta = np.zeros(L*q + L**2*q**2)

        #run the function
        log_PL = potts_log_pseudolikelihood(theta, seqs, L, q)

        #expected output
        #expected_log_PL = -np.log(2) * L
        expected = np.log(1/2) * 2

        #compare the output with the expected output
        assert np.all(log_PL == expected)

if __name__ == '__main__':
    unittest.main()
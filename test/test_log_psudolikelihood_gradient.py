import unittest
import numpy as np
from seqs_prob.log_pseudolikelihood_gradient import log_pseudolikelihood_gradient
from seqs_prob.potts_energy import potts_energy
from seqs_prob.potts_energy_gradient import potts_energy_gradient

#gradient = sum_n sum_p [dE(true letter) - sum_a P(a | x_-p) dE(x_p = a)]

class test_log_pseudolikelihood_gradient(unittest.TestCase):

    def test_log_pseudolikelihood_gradient(self):

        #input
        L = 2
        q = 2
        h = np.log(np.array([[1, 2],
                             [1, 3]]))
        J = np.zeros((L, L, q, q))
        theta = np.concatenate([h.flatten(), J.flatten()])
        encoded_sequences = np.array([[0, 1]])

        #potts energy and its gradient for log_pseudolikelihood_gradient
        def energy(seqs, theta):
            return potts_energy(seqs, theta, L, q)

        def energy_gradient(seqs):
            return potts_energy_gradient(seqs, L, q)

        #run the function
        gradient = log_pseudolikelihood_gradient(theta, energy, energy_gradient, encoded_sequences, q)

        #expected output
        #parameters used by each sequence:
        #  [0,0]: h_1(0), h_2(0), J_12(0,0)
        #  [0,1]: h_1(0), h_2(1), J_12(0,1)
        #  [1,1]: h_1(1), h_2(1), J_12(1,1)
        #position 1: true [0,1]; [0,1]: P = 1/3, [1,1]: P = 2/3
        #  h_1(0): 1 - 1/3 = 2/3        h_1(1): -2/3
        #  h_2(1): 1 - 1/3 - 2/3 = 0    
        #  J_12(0,1): 1 - 1/3 = 2/3     J_12(1,1): -2/3
        #position 2: true [0,1]; [0,0]: P = 1/4, [0,1]: P = 3/4
        #  h_1(0): 1 - 1/4 - 3/4 = 0    
        #  h_2(0): -1/4                 h_2(1): 1 - 3/4 = 1/4
        #  J_12(0,0): -1/4              J_12(0,1): 1 - 3/4 = 1/4
        #total (position 1 + position 2):
        #  h_1(0) = 2/3, h_1(1) = -2/3, h_2(0) = -1/4, h_2(1) = 1/4
        #  J_12(0,0) = -1/4, J_12(0,1) = 2/3 + 1/4 = 11/12, J_12(1,0) = 0, J_12(1,1) = -2/3
        #           h_1(0) h_1(1) h_2(0) h_2(1)
        expected = np.array([2/3, -2/3, -1/4, 1/4,
        #           J[0,0]: not used
                             0, 0, 0, 0,
        #           J[0,1]: (0,0) (0,1) (1,0) (1,1)
                             -1/4, 11/12, 0, -2/3,
        #           J[1,0]: not used
                             0, 0, 0, 0,
        #           J[1,1]: not used
                             0, 0, 0, 0])

        #compare the output with the expected output
        np.testing.assert_allclose(gradient, expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
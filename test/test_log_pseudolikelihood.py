import unittest
import numpy as np
from seqs_prob.log_pseudolikelihood import log_pseudolikelihood, log_pseudolikelihood_gradient
from seqs_prob.potts_model import potts_energy, potts_energy_gradient

#log PL = sum_n sum_p [E(true letter in sequence) - log sum_a exp(E(x_p = a))]

class test_log_pseudolikelihood(unittest.TestCase):

    def test_log_pseudolikelihood(self):

        #input
        L = 2
        q = 2
        h = np.log(np.array([[1, 2],
                             [1, 3]]))
        J = np.zeros((L, L, q, q))
        theta = np.concatenate([h.flatten(), J.flatten()])
        encoded_sequences = np.array([[0, 1]])

        #potts energy for log_pseudolikelihood
        def energy(seqs, theta):
            return potts_energy(seqs, theta, L, q)

        #run the function
        log_PL = log_pseudolikelihood(theta, energy, encoded_sequences, q)



        #P(true letter in sequence|other positions with all possible allels)=
        #   =e^E(true letter in sequence)/sum_allel e^E(this position with different allels)
        #expected output(from prior test)
        #e^E: [0,0] = 1, [0,1] = 3, [1,0] = 2, [1,1] = 6
        #position 1 (position 2 stays 1): compare [0,1] (true) and [1,1]
        #   e^E = 3 and 6  -->  3 / (3 + 6) = 1/3
        #position 2 (position 1 stays 0): compare [0,0] and [0,1] (true)
        #   e^E = 1 and 3  -->  3 / (1 + 3) = 3/4
        expected = np.log(1/3) + np.log(3/4)

        #compare the output with the expected output
        np.testing.assert_allclose(log_PL, expected)



    #gradient = sum_n sum_p [dE(true letter) - sum_a P(a | x_-p) dE(x_p = a)]

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
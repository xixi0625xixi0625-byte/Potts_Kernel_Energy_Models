import unittest
import numpy as np
from seqs_prob.log_pseudolikelihood import log_pseudolikelihood
from seqs_prob.potts_energy import potts_energy

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


if __name__ == '__main__':
    unittest.main()
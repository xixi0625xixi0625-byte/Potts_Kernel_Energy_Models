import unittest
import numpy as np
from seqs_prob.log_likelihood import log_likelihood
from seqs_prob.potts_energy import potts_energy

#logp = sum_n E(X_n) - N*logZ
#need: E(X_n): energy for every sequences
#      every possible sequences E


class test_log_likelihood(unittest.TestCase):

    def test_log_likelihood(self):

        #input
        L = 2
        q = 2
        h = np.log(np.array([[1, 2],
                             [1, 3]]))
        J = np.zeros((L, L, q, q))
        theta = np.concatenate([h.flatten(), J.flatten()])
        all_sequences = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        encoded_sequences = np.array([[0, 1], [1, 1]])

        #potts energy for log_likelihood
        def energy(seqs, theta):
            return potts_energy(seqs, theta, L, q)

        #run the function
        log_p = log_likelihood(theta, energy, encoded_sequences, all_sequences)

        #expected output:
        #e^E = e^h1(x1) * e^h2(x2)
        #[0, 0]: 1*1=1
        #[0, 1]: 1*3=3
        #[1, 0]: 2*1=2
        #[1, 1]: 2*3=6
        #Z = 1+3+2+6=12
        #p([0, 1])=3/12
        #p([1, 1])=6/12
        expected = np.log(3/12)+np.log(6/12)

        #comapre the output with the expected output
        np.testing.assert_allclose(log_p, expected, atol=1e-12)


    def test_log_likelihood_length_equals_three(self):
        
        #input
        L = 3
        q = 3
        h = np.log(np.array([[1, 1, 1],
                      [1, 2, 3],
                      [2, 3, 2]]))
        J = np.zeros((L, L, q, q))
        theta = np.concatenate([h.flatten(), J.flatten()])
        all_sequences = np.array([[1, 1, 1], [1, 1, 2], [1, 1, 0], 
                                  [1, 2, 1], [1, 2, 2], [1, 2, 0],
                                  [1, 0, 1], [1, 0, 2], [1, 0, 0],
                                  [2, 1, 1], [2, 1, 2], [2, 1, 0],
                                  [2, 2, 1], [2, 2, 2], [2, 2, 0],
                                  [2, 0, 1], [2, 0, 2], [2, 0, 0],
                                  [0, 1, 1], [0, 1, 2], [0, 1, 0],
                                  [0, 2, 1], [0, 2, 2], [0, 2, 0],
                                  [0, 0, 1], [0, 0, 2], [0, 0, 0]])
        encoded_sequences = np.array([[1, 2, 0], [0, 2, 0], [0, 0, 0]])

        #potts_energy for log_likelihood
        def energy(seqs, theta):
            return potts_energy(seqs, theta, L, q)

        #run the function 
        log_p2 = log_likelihood(theta, energy, encoded_sequences, all_sequences)

        #expected ouput:
        #e^E = e^h1(x1) * e^h2(x2)
        #[1, 1, 1]=6, [1, 1, 2]=4, [1, 1, 0]=4, 
        #[1, 2, 1]=9, [1, 2, 2]=6, [1, 2, 0]=6,
        #[1, 0, 1]=3, [1, 0, 2]=2, [1, 0, 0]=2,
        #[2, 1, 1]=6, [2, 1, 2]=4, [2, 1, 0]=4,
        #[2, 2, 1]=9, [2, 2, 2]=6, [2, 2, 0]=6,
        #[2, 0, 1]=3, [2, 0, 2]=2, [2, 0, 0]=2,
        #[0, 1, 1]=6, [0, 1, 2]=4, [0, 1, 0]=4,
        #[0, 2, 1]=9, [0, 2, 2]=6, [0, 2, 0]=6,
        #[0, 0, 1]=3, [0, 0, 2]=2, [0, 0, 0]=2
        #p[1, 2, 0]=6/126, p[0, 2, 0]=6/126, p[0, 0, 0]=2/126
        expected2 = np.log(6/126)+np.log(6/126)+np.log(2/126)

        #compare the output with the expected output
        np.testing.assert_allclose(log_p2, expected2, atol=1e-12)

if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
from seqs_prob.log_likelihood import log_likelihood, log_likelihood_gradient
from seqs_prob.potts_model import potts_energy, potts_energy_gradient

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



    def test_log_likelihood_gradient(self):
    
        #input
        L = 2
        q = 2
        h = np.log(np.array([[1, 2],
                                [1, 3]]))
        J = np.zeros((L, L, q, q))
        theta = np.concatenate([h.flatten(), J.flatten()])
        all_sequences = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        encoded_sequences = np.array([[0, 1], [1, 1]]) 

        #potts energy & its gradient for log_likelihood_gradient
        def energy(seqs, theta):
            return potts_energy(seqs, theta, L, q)

        def energy_gradient(seqs):
            return potts_energy_gradient(seqs, L, q)

        #run the function
        gradient = log_likelihood_gradient(theta, energy, energy_gradient, encoded_sequences, all_sequences)


        #from last function test: p([0, 0])=1/12; p([0, 1])=3/12; p([1, 0])=2/12; p([1, 1])=6/12
        #N=2


        #expected output
        #have N=2 (2 encoded sequencesz)
        #for h(4 parameters):
        #h1(0) (0 on the first position): data has 0 on the first position: only contains [0, 1]; 
        #                                 Model expected: [0, 0](1/12) + [0, 1](3/12) = 4/12
        #   gradient = 1 - 2*4/12 = 4/12
        #h1(1) (1 on the first position): data has 1 on the first position: only contains [1, 1];
        #                                 Model expected: [1, 0](2/12) + [1, 1](6/12) = 8/12
        #   gradient = 1 - 2*8/12 = -4/12
        #h2(0) (0 on the second position): data has 0 on the second position: None
        #                                 Model expected: [0, 0](1/12) + [1, 0](2/12) = 3/12
        #   gradient = 0 - 2*3/12 = -6/12
        #h2(1) (1 on the second position): data has 1 on the second position: contains [0, 1], [1, 1]
        #                                 Model expected: [0, 1](3/12) + [1, 1](6/12) = 9/12
        #   gradient = 2 - 2*9/12 = 6/12

        #for J(4 parameters):
        #J_12(0, 0): data has None; Model expected: [0, 0]: 1/12; gradient = 0-2*1/12=-1/6
        #J_12(0, 1): date has [0, 1]; Model expected: [0, 1]: 3/12; gradient = 1-2*3/12=6/12 
        #J_12(1, 0): data has None; Model expected: [1, 0]: 2/12; gradient = 0-2*2/12=-4/12
        #J_12(1, 1): data has [1, 1]; Model expected: [1, 1]: 6/12; gradient = 1-2*6/12=0
        
        #           h_1(0) h_1(1) h_2(0) h_2(1)
        expected = np.array([1/3, -1/3, -1/2, 1/2,
        #           J[0,0]: not used
                                0, 0, 0, 0,
        #           J[0,1]: (0,0) (0,1) (1,0) (1,1)
                                -1/6, 1/2, -1/3, 0,
        #           J[1,0]: not used
                                0, 0, 0, 0,
        #           J[1,1]: not used
                                0, 0, 0, 0])

        #compare the output with the expected output
        np.testing.assert_allclose(gradient, expected, atol=1e-12)

if __name__ == '__main__':
    unittest.main()
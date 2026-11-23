import unittest
import numpy as np
from seqs_prob.log_likelihood_gradient import log_likelihood_gradient
from seqs_prob.potts_energy import potts_energy
from seqs_prob.potts_energy_gradient import potts_energy_gradient

#gradient = sum_n dE(x_n) - N * sum_x p(x) dE(x)
#for potts: gradient = (count in the data) - N * (probability in the model)

class test_log_likelihood_gradient(unittest.TestCase):

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
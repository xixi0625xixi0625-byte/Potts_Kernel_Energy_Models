import unittest
import numpy as np
from seqs_prob.potts_model import potts_energy_gradient, potts_conditional_energy, potts_energy, potts_log_pseudolikelihood, potts_prior_gradient, potts_prior

class test_potts_model(unittest.TestCase):

    def test_potts_energy_gradient(self):

        #input
        seqs = np.array([[0, 1]])
        L = 2
        q = 2

        #run the function
        gradient = potts_energy_gradient(seqs, L, q)

        #expected output: sequence [0, 1] uses h_1(0), h_2(1), J_12(0,1)
        #             h_1(0) h_1(1) h_2(0) h_2(1)
        expected = np.array([1, 0, 0, 1,
        #             J[0,0]: not used
                             0, 0, 0, 0,
        #             J[0,1]: (0,0) (0,1) (1,0) (1,1)
                             0, 1, 0, 0,
        #             J[1,0]: not used
                             0, 0, 0, 0,
        #             J[1,1]: not used
                             0, 0, 0, 0])
        
        #compare the output with the expected output
        assert np.all(gradient == expected)


    def test_potts_conditional_energy(self):
    
        #input
        seqs = np.array([[0, 1]])
        L = 2
        q = 2
        h = np.array([[1, 2],      #h_1(0), h_1(1)
                        [3, 4]])     #h_2(0), h_2(1)
        J = np.zeros((L, L, q, q))
        J[0, 1] = np.array([[1, 2],    #J_12(0,0), J_12(0,1)
                            [3, 4]])   #J_12(1,0), J_12(1,1)
        theta = np.concatenate([h.flatten(), J.flatten()])

        #run the function
        E_conditional = potts_conditional_energy(seqs, theta, L, q)

        #expected output
        #position 1, position 2 is fixed at x_2 = 1:
        #  E(a | x_-1) = h_1(a) + J_12(a, 1)
        #  a = 0: 1 + 2 = 3
        #  a = 1: 2 + 4 = 6
        #position 2, position 1 is fixed at x_1 = 0:
        #  E(a | x_-2) = h_2(a) + J_12(0, a)
        #  a = 0: 3 + 1 = 4
        #  a = 1: 4 + 2 = 6
        expected = np.array([[[3.0, 6.0],      #position 1, a = 0, 1
                                [4.0, 6.0]]])    #position 2, a = 0, 1

        print(E_conditional)
        print(expected)
        print(J)

        #compare the output with the expected output
        assert np.all(E_conditional == expected)


    def test_potts_energy(self):
    
        #input
        seqs = np.array([[0, 0], [1, 1], [0, 0]])
        L = 2
        q = 2
        theta = np.zeros((L*q + L**2*q**2))

        #run the function
        energy = potts_energy(seqs, theta, L, q)

        #expected output: 0
        expected = np.array([0, 0, 0])

        #compare the output with the expected output
        assert np.all(energy == expected)
    
    
    def test_potts_energy_complex(self):
    
        #input
        seqs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        L =2
        q =2
        #h: L * q
        #h_1(0) = 1, h_1(1) = 2   grade!
        #h_2(0) = 3, h_2(1) = 4
        h = np.array([[1, 2], 
                        [3, 4]])
        #J: L * L * q * q
        #J_12(0, 0) = 10, J_12(0, 1) = 20
        #J_12(1, 0) = 30, J_12(1, 1) = 40
        J = np.zeros((L, L, q, q))
        J[0, 1] = np.array([[10, 20],
                            [30, 40]])

        theta = np.concatenate([h.flatten(), J.flatten()])

        #run the function
        energy = potts_energy(seqs, theta, L, q)

        #expected output
        #E(x) = h_1(x_1) + h_2(x_2) + J_12(x_1, x_2)
        #[0, 0]: 1 + 3 + 10 = 14
        #[0, 1]: 1 + 4 + 20 = 25
        #[1, 0]: 2 + 3 + 30 = 35
        #[1, 1]: 2 + 4 + 40 = 46
        expected = np.array([1 + 3 + 10, 1 + 4 + 20, 2 + 3 + 30, 2 + 4 + 40])

        #compare the output with the expected output
        assert np.all(energy == expected)

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


    def test_potts_prior(self):
    
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
        prior = potts_prior(theta, L, q, sigma_h_squared, sigma_J_squared)

        #expected output
        #-sum h^2 / (2 sigma_h^2) - sum J^2 / (2 sigma_J^2)
        #sum h^2 = 1^2 + 2^2 + 3^2 + 4^2 = 30     -30 / (2*2) = -7.5
        #sum J^2 = 1^2 + 2^2 = 5        -5 / (2*1) = -2.5
        #-7.5 - 2.5 = -10
        expected = -10

        #compare the output with the expected output
        assert np.all(prior == expected)
    

if __name__ == '__main__':
    unittest.main()
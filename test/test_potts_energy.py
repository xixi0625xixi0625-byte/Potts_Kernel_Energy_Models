import unittest
import numpy as np
from seqs_prob.potts_energy import potts_energy

class test_potts_model(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()



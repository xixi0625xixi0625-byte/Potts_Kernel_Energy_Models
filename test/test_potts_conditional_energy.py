import unittest
import numpy as np
from seqs_prob.potts_conditional_energy import potts_conditional_energy

class test_potts_conditional_energy(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
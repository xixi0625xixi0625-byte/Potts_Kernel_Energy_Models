import unittest
import numpy as np
from seqs_prob.potts_energy_gradient import potts_energy_gradient

class test_potts_energy_gradient(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
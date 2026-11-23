import unittest
import numpy as np
from seqs_prob.kernel_energy import kernel_energy

class test_kernel_energy(unittest.TestCase):

    def test_kernel_energy(self):

        #input
        seqs = np.array([[0, 0], [0, 1]])
        U = np.array([[0, 0], [1, 1]])
        alpha = np.array([1, 2])
        sigma_E_squared =1
        c = 1

        #run the function
        energy = kernel_energy(seqs, alpha, U, sigma_E_squared, c)

        #expected output
        #E(x) = K(x, u_1)*alpha1 + K(x, u_2)*alpha2
        #k = exp(-H)
        #[0, 0]: e^0*1+e^-2*2=1+2e^-2
        #[0, 1]: e^-1*1+e^-1*2=3e^-1

        expected = np.array([1+2*np.exp(-2), 3*np.exp(-1)])

        #compare the output with the expected output
        assert np.all(energy == expected)

if __name__ == '__main__':
    unittest.main()
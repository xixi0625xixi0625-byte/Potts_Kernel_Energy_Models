import unittest
import numpy as np
from seqs_prob.hamming_distance import hamming_distance

class test_hamming_distance(unittest.TestCase):

    def test_hamming_distance(self):

        #input
        seqs = np.array([[0, 1, 2, 3], #ACGT
                         [0, 1, 2, 0], #ACGA
                         [3, 3, 3, 3]])#TTTT

        #run the function
        H = hamming_distance(seqs, seqs)

        #expected output
        expected = np.array([[0, 1, 3], #expected(1, 2): difference between the first seq and the second seq (ACGT vs ACGA) is 1
                             [1, 0, 4],
                             [3, 4, 0]])

        #compare the output with the expected output
        assert np.all(H == expected)

if __name__ == '__main__':
    unittest.main()


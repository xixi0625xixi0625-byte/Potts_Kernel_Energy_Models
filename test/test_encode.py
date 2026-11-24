import unittest
import numpy as np
from seqs_prob.encode import encode

class test_encode(unittest.TestCase):

    def test_encode(self):

        #input: sequences and alphabet
        seqs = ["ACGT", "TTAA"]
        alphabet = "ACGT"

        #output: encoded sequences
        #run the function
        encoded = encode(seqs, alphabet)

        #expected output
        expected = np.array([[0, 1, 2, 3], [3, 3, 0, 0]])

        #compare the output with the expected output
        assert np.all(encoded == expected)


if __name__ == '__main__':
    unittest.main()
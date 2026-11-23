import unittest
import numpy as np
from seqs_prob.get_all_sequences import get_all_sequences

class test_get_all_sequences(unittest.TestCase):

    def test_small_example(self):

        #input
        L =2
        q = 2

        #expected output
        expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        #run the function
        all_sequences = get_all_sequences(L, q)

        #compare the output with the expected output
        assert np.all(all_sequences == expected)

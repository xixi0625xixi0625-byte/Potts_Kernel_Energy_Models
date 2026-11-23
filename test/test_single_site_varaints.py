import unittest
import numpy as np
from seqs_prob.single_site_variants import single_site_variants

class test_single_site_variants(unittest.TestCase):

    def test_single_site_variants(self):

        #input
        x = np.array([0, 1, 1])
        p = 1
        q = 3

        #run the function
        variants = single_site_variants(x, p, q)

        #expected output: position 1 is changing to 0, 1, 2, the other positions fixed
        expected = np.array([[0, 0, 1],
                             [0, 1, 1],
                             [0, 2, 1]])

        #compare the output with the expected output
        assert np.all(expected == variants)

if __name__ == '__main__':
    unittest.main()
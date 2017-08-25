import unittest
import tensorflow as tf
from networks import *

class NetworkTests(unittest.TestCase):

    def testConstructSizes(self):
        nn = FFNet([2,3,1], 1)
        nn.construct()
        self.assertEqual(len(nn.weights), 2, "Number of weight matrices incorrect")
        self.assertEqual(len(nn.biases), 2, "Number of bias vectors incorrect")
        self.assertEqual(len(nn.layer_calcs), 2, "Number of layer calculation vectors incorrect")
        self.assertEqual(len(nn.layer_activations), 2, "Number of layer activation vectors incorrect")

def main():
    unittest.main()

if __name__ == '__main__':
    main()

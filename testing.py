import unittest

import dataset

class TestingDataset(unittest.TestCase):

    def test_load(self):
        E13 = dataset.Dataset("datasets/human/E13")
        TFP = dataset.Dataset("datasets/human/TFP")
        FSF = dataset.Dataset("datasets/fake/FSF")
        INT = dataset.Dataset("datasets/fake/INT")
        TWT = dataset.Dataset("datasets/fake/TWT")

        # based on paper numbers
        self.assertEqual(TFP.size, 469)
        self.assertEqual(E13.size, 1481)
        self.assertEqual(FSF.size, 1169)
        self.assertEqual(INT.size, 1337)
        self.assertEqual(TWT.size, 845)

if __name__ == "__main__":
    unittest.main()


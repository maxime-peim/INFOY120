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

    def test_union(self):
        E13 = dataset.Dataset("datasets/human/E13")
        TFP = dataset.Dataset("datasets/human/TFP")

        HUM = E13 + TFP
        self.assertEqual(HUM.size, 1950)

    def test_subset(self):
        E13 = dataset.Dataset("datasets/human/E13")
        TFP = dataset.Dataset("datasets/human/TFP")

        HUM = E13 + TFP
        self.assertLess(E13, HUM)
        self.assertGreater(HUM, TFP)

    def test_equal(self):
        E13_1 = dataset.Dataset("datasets/human/E13")
        E13_2 = dataset.Dataset("datasets/human/E13")

        self.assertEqual(E13_1, E13_2)
        self.assertLessEqual(E13_1, E13_2)
        self.assertGreaterEqual(E13_1, E13_2)

    def test_undersampling(self):
        FSF = dataset.Dataset("datasets/fake/FSF")
        INT = dataset.Dataset("datasets/fake/INT")
        TWT = dataset.Dataset("datasets/fake/TWT")

        num_points = 1950

        FAK = FSF + INT + TWT
        FAK.inplace_undersample(num_points)

        self.assertEqual(FAK.size, num_points)

        FAK = FSF + INT + TWT
        UFAK = FAK.undersample(num_points)

        self.assertIsNot(UFAK, FAK)
        self.assertEqual(UFAK.size, num_points)
        

if __name__ == "__main__":
    unittest.main()


import os
import unittest
from settings import Settings
from dataloader_iam import JPSDSmallTestSet

class TestJPSDSmallDataLoader(unittest.TestCase):

    def test_number_of_samples_should_be_same_number_of_jpg_files(self):
        # Arrange
        dataset_loader = JPSDSmallTestSet(Settings.JPSD_SMALL_PATH)
        files = []
        for filename in os.listdir(Settings.JPSD_SMALL_PATH + "/lines"):
            if filename.endswith('.jpg'):
                files.append(filename)
        # Act
        samples = dataset_loader._load_samples()

        # Assert
        self.assertEqual(len(samples), len(files))

if __name__ == "__main__":
    unittest.main()
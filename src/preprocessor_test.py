import cv2
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

class PreprocessorTest():
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def test_random_preprocess_image(self):
        img = cv2.imread('../data/line.png', cv2.IMREAD_GRAYSCALE)
        img_aug = self.preprocessor.process_img(img)
        img_aug = cv2.transpose(img_aug) + 0.5
        visualize(img, img_aug)

def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(2,1,1)
  plt.title('Original image')
  plt.imshow(original, cmap='gray')

  plt.subplot(2,1,2)
  plt.title('Augmented image')
  plt.imshow(augmented, cmap='gray', vmin=0, vmax=1)
  plt.show()

def main():
    preprocessor = Preprocessor((256, 32), data_augmentation=True)
    test = PreprocessorTest(preprocessor)
    test.test_random_preprocess_image()

if __name__ == '__main__':
    main()
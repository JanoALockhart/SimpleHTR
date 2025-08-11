import numpy
import cv2
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

class PreprocessorTest():
    preprocessor: Preprocessor
    img_path: str

    def __init__(self, preprocessor, img_path = '../data/line.png'):
        self.preprocessor = preprocessor
        self.img_path = img_path

    def _open_image(self):
        return cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)

    def test_random_preprocess_image(self):
        original_img = self._open_image()

        img_aug = self.preprocessor.process_img(original_img)
        img_aug = cv2.transpose(img_aug) + 0.5
        
        self._visualize(original_img, img_aug)

    def _visualize(self, original, augmented):
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.title('Original image')
        plt.imshow(original, cmap='gray')

        plt.subplot(2,1,2)
        plt.title('Augmented image')
        plt.imshow(augmented, cmap='gray')
        plt.show()

def main():
    preprocessor = Preprocessor((256, 32), data_augmentation=True)
    test = PreprocessorTest(preprocessor)
    test.test_random_preprocess_image()
    #test.test_gaussian_blur()

if __name__ == '__main__':
    main()
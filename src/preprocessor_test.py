import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

class PreprocessorTest():
    preprocessor: Preprocessor
    img_path: str

    def __init__(self, preprocessor, img_path = '../data/line.png'):
        self.preprocessor = preprocessor
        self.img_path = img_path

    def _open_image(self) -> np.ndarray:
        return cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)

    def test_random_preprocess_image(self):
        original_img = self._open_image()

        img_aug = self.preprocessor.process_img(original_img)
        img_aug = cv2.transpose(img_aug) + 0.5
        
        self._visualize(original_img, img_aug, "Combined Techniques")
    
    def test_gaussian_blur(self):
        original_img = self._open_image()

        kernels = [(3, 3), (3, 5), (3, 7),
                   (5, 3), (5, 5), (5, 7),
                   (7, 3), (7 ,5), (7, 7),
                   ]
        
        for kernel in kernels:
            augmented_img = self.preprocessor.gaussian_blur(original_img, kernel)
            self._visualize(original_img, augmented_img, "Gaussian Blur. Kernel=" + str(kernel))

    def test_dilate(self):
        original_img = self._open_image()

        augmented_img = self.preprocessor.dilate(original_img)

        self._visualize(original_img, augmented_img, "Dilate")

    def test_erode(self):
        original_img = self._open_image()

        augmented_img = self.preprocessor.erode(original_img)

        self._visualize(original_img, augmented_img, "Erode")

    def test_random_transformation(self):
        original_img = self._open_image().astype(np.float)

        augmented_img = self.preprocessor.random_transformation(original_img)

        self._visualize(original_img, augmented_img, "Random Transformation")

    def _visualize(self, original, augmented, title = ""):
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.title('Original image')
        plt.imshow(original, cmap='gray')

        plt.subplot(2,1,2)
        plt.title('Augmented image')
        plt.imshow(augmented, cmap='gray')

        plt.suptitle(title)

        plt.show()

def main():
    preprocessor = Preprocessor((256, 32), data_augmentation=True)
    test = PreprocessorTest(preprocessor)
    #test = PreprocessorTest(preprocessor, '../data/word.png')
    
    #test.test_random_preprocess_image()
    #test.test_gaussian_blur()
    #test.test_dilate()
    #test.test_erode()
    test.test_random_transformation()

if __name__ == '__main__':
    main()
    
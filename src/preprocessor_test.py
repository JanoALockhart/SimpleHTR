import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocessor import Preprocessor, ColorImagePreprocessor

class PreprocessorTest():
    preprocessor: Preprocessor
    img_path: str

    def __init__(self, preprocessor: Preprocessor, img_path = '../data/line.png'):
        self.preprocessor = preprocessor
        self.img_path = img_path

    def _open_image(self) -> np.ndarray:
        return cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
    
    def show_original_image(self):
        original_img = self._open_image()
        self._visualize(original_img, original_img)

    def test_random_preprocess_image(self):
        original_img = self._open_image()

        img_aug = self.preprocessor.process_img(original_img)
        img_aug = cv2.transpose(img_aug) + 0.5
        
        self._visualize(original_img, img_aug, title="Combined Techniques", normalized=True)
    
    def test_gaussian_blur(self):
        original_img = self._open_image()

        kernels = [(3, 3), (3, 5), (3, 7),
                   (5, 3), (5, 5), (5, 7),
                   (7, 3), (7 ,5), (7, 7),
                   ]
        
        for kernel in kernels:
            augmented_img = self.preprocessor._gaussian_blur(original_img, kernel)
            self._visualize(original_img, augmented_img, "Gaussian Blur. Kernel=" + str(kernel))

    def test_dilate(self):
        original_img = self._open_image()

        augmented_img = self.preprocessor._dilate(original_img)

        self._visualize(original_img, augmented_img, "Dilate")

    def test_erode(self):
        original_img = self._open_image()

        augmented_img = self.preprocessor._erode(original_img)

        self._visualize(original_img, augmented_img, "Erode")

    def test_random_transformation(self):
        original_img = self._open_image().astype(np.float)

        for i in range(3):
            augmented_img = self.preprocessor._random_transformation(original_img)
            self._visualize(original_img, augmented_img, "Random Transformation")
    
    def test_random_transformation_scaling_multiplier(self, scaling_multiplier):
        original_img = self._open_image().astype(np.float)

        augmented_img = self.preprocessor._random_transformation(original_img, scaling_multiplier, scaling_multiplier)

        self._visualize(original_img, augmented_img, "Random Transformation. Scaling Multiplier = " + str(scaling_multiplier))

    def test_darken(self):
        original_img = self._open_image().astype(np.float)

        darkening_factors = (0.25, 0.5, 0.75, 1)
        for darkening_factor in darkening_factors:
            augmented_img = self.preprocessor._darken(original_img, darkening_factor)
            self._visualize(original_img, augmented_img, "Darkening. Darkening Factor = " + str(darkening_factor))

    def test_noise(self):
        original_img = self._open_image().astype(np.float)

        noise_multiplier = 25
        augmented_img = self.preprocessor._noise(original_img, noise_multiplier = noise_multiplier)

        self._visualize(original_img, augmented_img, "Random Transformation. Noise Multiplier = " + str(noise_multiplier))

    def test_invert(self):
        original_img = self._open_image().astype(np.float)

        augmented_img = self.preprocessor._invert(original_img)

        self._visualize(original_img, augmented_img, "Invert")

    def test_transpose(self):
        original_img = self._open_image().astype(np.float)

        augmented_img = self.preprocessor._transpose(original_img)

        self._visualize(original_img, augmented_img, "Transpose")
    
    def test_normalize(self):
        original_img = self._open_image().astype(np.float)
        print(np.min(original_img))
        print(np.max(original_img))
        augmented_img = self.preprocessor._normalize(original_img)
        print(np.min(augmented_img))
        print(np.max(augmented_img))
        assert -0.5 <= np.min(augmented_img) and np.max(augmented_img) <= 0.5

    def _visualize(self, original, augmented, title = "", normalized = False):
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.title('Original image')
        plt.imshow(original, cmap='gray')

        plt.subplot(2,1,2)
        plt.title('Augmented image')
        vmax = 255
        if normalized:
            vmax=1
        plt.imshow(augmented, cmap='gray', vmin=0, vmax=vmax)

        plt.suptitle(title)

        plt.show()

class ColorImagePreprocessorTest(PreprocessorTest):
    def __init__(self, preprocessor: ColorImagePreprocessor, img_path='../data/line.png'):
        super().__init__(preprocessor, img_path)
        self.preprocessor2 = preprocessor

    def test_thresholding(self):
        original_img = super()._open_image()

        aug_img = self.preprocessor2._thresholding(original_img)

        assert np.min(aug_img) == 0 and np.max(aug_img) == 255
        self._visualize(original_img, aug_img, "Threshold")

def main():
    preprocessor = Preprocessor(data_augmentation=True, line_mode=True)
    #img_path = '../data/word.png'
    img_path = "C:\\Users\\janoa\\Documents\\UNS\\TRABAJO FINAL\\Datasets\\JPSD-small\\lines\\n001-00-01.jpg"
    test = PreprocessorTest(preprocessor, img_path)

    #test.test_random_preprocess_image()
    #test.test_gaussian_blur()
    #test.test_dilate()
    #test.test_erode()
    #test.test_random_transformation()
    #test.test_random_transformation_scaling_multiplier(scaling_multiplier=0.75)
    #test.test_random_transformation_scaling_multiplier(scaling_multiplier=1.05)
    #test.test_darken()
    #test.test_noise()
    #test.test_invert()
    #test.test_transpose()
    #test.test_normalize()

    preprocessor = ColorImagePreprocessor(line_mode=True)
    test = ColorImagePreprocessorTest(preprocessor, img_path)

    test.test_thresholding()


if __name__ == '__main__':
    main()
    
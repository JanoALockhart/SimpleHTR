import cv2
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

def test_random_preprocess_image():
    img = cv2.imread('../data/line.png', cv2.IMREAD_GRAYSCALE)
    img_aug = Preprocessor((256, 32), data_augmentation=True).process_img(img)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(cv2.transpose(img_aug) + 0.5, cmap='gray', vmin=0, vmax=1)
    plt.show()

def main():
    test_random_preprocess_image()

if __name__ == '__main__':
    main()
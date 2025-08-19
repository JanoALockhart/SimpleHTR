import pickle
import random
from typing import Tuple

import cv2
import lmdb
import numpy as np
from path import Path

from dataset_structure import Batch

def get_img_height() -> int:
    """Fixed height for NN."""
    return 32

def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


class Preprocessor:
    def __init__(self,
                 data_dir: Path,
                 padding: int = 0,
                 dynamic_width: bool = False,
                 data_augmentation: bool = False,
                 line_mode: bool = False,
                 fast: bool = True) -> None:
        # dynamic width only supported when no data augmentation happens
        assert not (dynamic_width and data_augmentation)
        # when padding is on, we need dynamic width enabled
        assert not (padding > 0 and not dynamic_width)

        self.target_img_size = get_img_size()
        self.padding = padding
        self.dynamic_width = dynamic_width
        self.data_augmentation = data_augmentation
        self.line_mode = line_mode

    @staticmethod
    def _truncate_label(text: str, max_text_len: int) -> str:
        """
        Function ctc_loss can't compute loss if it cannot find a mapping between text label and input
        labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        If a too-long label is provided, ctc_loss returns an infinite gradient.
        """
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def _simulate_text_line(self, batch: Batch) -> Batch:
        """Create image of a text line by pasting multiple word images into an image."""

        default_word_separation = 30
        default_words_in_line = 5

        # go over all batch elements
        result_imgs = []
        result_groundtruth_texts = []
        for i in range(batch.batch_size):
            # number of words to put into current line
            min_words_in_line = 1
            max_words_in_line = 8
            num_words_in_line = random.randint(min_words_in_line, max_words_in_line) if self.data_augmentation else default_words_in_line

            # concat ground truth texts
            current_groundtruth = ' '.join([batch.gt_texts[(i + j) % batch.batch_size] for j in range(num_words_in_line)])
            result_groundtruth_texts.append(current_groundtruth)

            # put selected word images into list, compute target image size
            selected_imgs = []
            word_separations = [0]
            target_height = 0
            target_width = 0
            for j in range(num_words_in_line):
                current_selected_img = batch.imgs[(i + j) % batch.batch_size]
                min_separation = 20
                max_separation = 50
                current_word_separation = random.randint(min_separation, max_separation) if self.data_augmentation else default_word_separation
                target_height = max(target_height, current_selected_img.shape[0])
                target_width += current_selected_img.shape[1]
                selected_imgs.append(current_selected_img)
                if j + 1 < num_words_in_line:
                    target_width += current_word_separation
                    word_separations.append(current_word_separation)

            # put all selected word images into target image
            target = np.ones([target_height, target_width], np.uint8) * 255
            x = 0
            for current_selected_img, current_word_separation in zip(selected_imgs, word_separations):
                x += current_word_separation
                y = (target_height - current_selected_img.shape[0]) // 2
                target[y:y + current_selected_img.shape[0]:, x:x + current_selected_img.shape[1]] = current_selected_img
                x += current_selected_img.shape[1]

            # put image of line into result
            result_imgs.append(target)

        return Batch(result_imgs, result_groundtruth_texts, batch.batch_size)

    def process_img(self, img: np.ndarray) -> np.ndarray:
        """Resize to target size, apply data augmentation."""

        # there are damaged files in IAM dataset - just use black image instead
        if img is None:
            img = np.zeros(self.target_img_size[::-1])

        img = img.astype(np.float)

        if self.data_augmentation:
            img = self._random_gaussian_blur(img)
            img = self._random_dilate(img)
            img = self._random_erode(img)

            img = self._random_transformation(img)

            img = self._random_darkening(img)
            img = self._random_noise(img)
            img = self._random_invert(img)

        else:
            if self.dynamic_width:
                img = self._dynamic_width_transformation(img)
            else:
                img = self._basic_scale_translate_transformation(img)

        img = self._transpose(img) # transpose for TF
        img = self._normalize(img)

        return img

    def _normalize(self, img):
        """Convert to range [-1, 1]. Specifically [-0.5, 0.5]"""
        img = img / 255 - 0.5
        return img

    def _transpose(self, img):
        img = cv2.transpose(img)
        return img

    def _random_invert(self, img):
        if random.random() < 0.1:
            img = self._invert(img)
        return img

    def _invert(self, img):
        img = 255 - img
        return img

    def _random_noise(self, img, probability = 0.25, max_noise_multiplier = 25):
        if random.random() < probability:
            random_noise_multiplier = random.randint(1, max_noise_multiplier)
            img = self._noise(img, random_noise_multiplier)
        return img

    def _noise(self, img: np.ndarray, noise_multiplier):
        noise = (np.random.random(img.shape) - 0.5) * noise_multiplier
        img = np.clip(img + noise, 0, 255)
        return img

    def _random_darkening(self, img, probability = 0.5, min_darkening_factor = 0.25):
        if random.random() < probability:
            random_darkening_factor = min_darkening_factor + random.random() * (1 - min_darkening_factor)
            img = self._darken(img, random_darkening_factor)
        return img

    def _darken(self, img, darkening_factor):
        img = img * darkening_factor
        return img

    def _basic_scale_translate_transformation(self, img: np.ndarray):
        target_width, target_height = self.target_img_size
        image_height, image_width = img.shape

        scaling_factor = min(target_width / image_width, target_height / image_height)
        translation_x = (target_width - image_width * scaling_factor) / 2
        translation_y = (target_height - image_height * scaling_factor) / 2

        transformation_matrix = np.float32([
                    [scaling_factor, 0, translation_x], 
                    [0, scaling_factor, translation_y]
                ])
                
        target_shape = (target_width, target_height)
        img = self._apply_transformation(img, transformation_matrix, target_shape)

        return img

    def _dynamic_width_transformation(self, img: np.ndarray):
        target_height = self.target_img_size[1]
        image_height, image_width = img.shape

        scaling_factor = target_height / image_height
        target_width = int(scaling_factor * image_width + self.padding)
        target_width = target_width + (4 - target_width) % 4
        translation_x = (target_width - image_width * scaling_factor) / 2
        translation_y = 0

        transformation_matrix = np.float32([
                    [scaling_factor, 0, translation_x], 
                    [0, scaling_factor, translation_y]
                ])
                
        target_shape = (target_width, target_height)
        img = self._apply_transformation(img, transformation_matrix, target_shape)

        return img

    def _random_transformation(self, img: np.ndarray, min_scaling_multiplier = 0.75, max_scaling_multiplier = 1.05):
        target_width, target_height = self.target_img_size
        image_height, image_width = img.shape

        scaling_factor = min(target_width / image_width, target_height / image_height)
        
        random_scaling_factor_x = scaling_factor * np.random.uniform(min_scaling_multiplier, max_scaling_multiplier)
        random_scaling_factor_y = scaling_factor * np.random.uniform(min_scaling_multiplier, max_scaling_multiplier)

        center_translation_x = (target_width - image_width * random_scaling_factor_x) / 2
        center_translation_y = (target_height - image_height * random_scaling_factor_y) / 2
        freedom_x = max((target_width - random_scaling_factor_x * image_width) / 2, 0)
        freedom_y = max((target_height - random_scaling_factor_y * image_height) / 2, 0)
        random_translation_x = center_translation_x + np.random.uniform(-freedom_x, freedom_x)
        random_translation_y = center_translation_y + np.random.uniform(-freedom_y, freedom_y)

        transformation_matrix = np.float32([
                    [random_scaling_factor_x, 0, random_translation_x], 
                    [0, random_scaling_factor_y, random_translation_y]
                ])
        
        target_shape = self.target_img_size
        img = self._apply_transformation(img, transformation_matrix, target_shape)

        return img

    def _apply_transformation(self, img, transformation_matrix, target_shape):
        target_image = np.ones(target_shape[::-1]) * 255
        img = cv2.warpAffine(img, transformation_matrix, dsize=target_shape, dst=target_image, borderMode=cv2.BORDER_TRANSPARENT)
        return img

    def _random_erode(self, img, probability = 0.25):
        if random.random() < probability:
            img = self._erode(img)
        return img

    def _erode(self, img):
        kernel = np.ones((3, 3))
        img = cv2.erode(img, kernel)
        return img

    def _random_dilate(self, img, probability = 0.25):
        if random.random() < probability:
            img = self._dilate(img)
        return img

    def _dilate(self, img):
        kernel = np.ones((3, 3))
        img = cv2.dilate(img, kernel)
        return img

    def _generate_random_odd_number(self):
        return random.randint(1, 3) * 2 + 1

    def _random_gaussian_blur(self, img, probability = 0.25):
        if random.random() < probability:
            random_kernel_size = (self._generate_random_odd_number(), self._generate_random_odd_number())
            img = self._gaussian_blur(img, random_kernel_size)
        return img

    def _gaussian_blur(self, img, kernel_size):
        img = cv2.GaussianBlur(img, kernel_size, 0)
        return img

    def process_batch(self, batch: Batch) -> Batch:
        if self.line_mode:
            batch = self._simulate_text_line(batch)

        res_imgs = [self.process_img(img) for img in batch.imgs]
        max_text_len = res_imgs[0].shape[0] // 4
        res_gt_texts = [self._truncate_label(gt_text, max_text_len) for gt_text in batch.gt_texts]
        return Batch(res_imgs, res_gt_texts, batch.batch_size)


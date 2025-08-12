import random
from typing import Tuple

import cv2
import numpy as np

from dataloader_iam import Batch


class Preprocessor:
    def __init__(self,
                 img_size: Tuple[int, int],
                 padding: int = 0,
                 dynamic_width: bool = False,
                 data_augmentation: bool = False,
                 line_mode: bool = False) -> None:
        # dynamic width only supported when no data augmentation happens
        assert not (dynamic_width and data_augmentation)
        # when padding is on, we need dynamic width enabled
        assert not (padding > 0 and not dynamic_width)

        self.img_size = img_size
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

        default_word_sep = 30
        default_num_words = 5

        # go over all batch elements
        res_imgs = []
        res_gt_texts = []
        for i in range(batch.batch_size):
            # number of words to put into current line
            num_words = random.randint(1, 8) if self.data_augmentation else default_num_words

            # concat ground truth texts
            curr_gt = ' '.join([batch.gt_texts[(i + j) % batch.batch_size] for j in range(num_words)])
            res_gt_texts.append(curr_gt)

            # put selected word images into list, compute target image size
            sel_imgs = []
            word_seps = [0]
            h = 0
            w = 0
            for j in range(num_words):
                curr_sel_img = batch.imgs[(i + j) % batch.batch_size]
                curr_word_sep = random.randint(20, 50) if self.data_augmentation else default_word_sep
                h = max(h, curr_sel_img.shape[0])
                w += curr_sel_img.shape[1]
                sel_imgs.append(curr_sel_img)
                if j + 1 < num_words:
                    w += curr_word_sep
                    word_seps.append(curr_word_sep)

            # put all selected word images into target image
            target = np.ones([h, w], np.uint8) * 255
            x = 0
            for curr_sel_img, curr_word_sep in zip(sel_imgs, word_seps):
                x += curr_word_sep
                y = (h - curr_sel_img.shape[0]) // 2
                target[y:y + curr_sel_img.shape[0]:, x:x + curr_sel_img.shape[1]] = curr_sel_img
                x += curr_sel_img.shape[1]

            # put image of line into result
            res_imgs.append(target)

        return Batch(res_imgs, res_gt_texts, batch.batch_size)

    def process_img(self, img: np.ndarray) -> np.ndarray:
        """Resize to target size, apply data augmentation."""

        # there are damaged files in IAM dataset - just use black image instead
        if img is None:
            img = np.zeros(self.img_size[::-1])

        # data augmentation
        img = img.astype(np.float)
        if self.data_augmentation:
            # photometric data augmentation
            img = self.random_gaussian_blur(img)
            img = self.random_dilate(img)
            img = self.random_erode(img)

            # geometric data augmentation
            img = self.random_transformation(img)
            
            # photometric data augmentation
            if random.random() < 0.5:
                img = img * (0.25 + random.random() * 0.75)
            if random.random() < 0.25:
                img = np.clip(img + (np.random.random(img.shape) - 0.5) * random.randint(1, 25), 0, 255)
            if random.random() < 0.1:
                img = 255 - img

        # no data augmentation
        else:
            if self.dynamic_width:
                img = self.dynamic_width_transformation(img)

            else:
                target_width, target_height = self.img_size
                image_height, image_width = img.shape

                scaling_factor = min(target_width / image_width, target_height / image_height)
                random_translation_x = (target_width - image_width * scaling_factor) / 2
                random_translation_y = (target_height - image_height * scaling_factor) / 2

                # map image into target image
                transformation_matrix = np.float32([
                    [scaling_factor, 0, random_translation_x], 
                    [0, scaling_factor, random_translation_y]
                ])
                
                target_shape = (target_width, target_height)
                img = self._apply_transformation(img, transformation_matrix, target_shape)

        # transpose for TF
        img = cv2.transpose(img)

        # convert to range [-1, 1]
        img = img / 255 - 0.5
        return img

    def dynamic_width_transformation(self, img):
        target_height = self.img_size[1]
        image_height, image_width = img.shape

        scaling_factor = target_height / image_height
        target_width = int(scaling_factor * image_width + self.padding)
        target_width = target_width + (4 - target_width) % 4
        random_translation_x = (target_width - image_width * scaling_factor) / 2
        random_translation_y = 0

                # map image into target image
        transformation_matrix = np.float32([
                    [scaling_factor, 0, random_translation_x], 
                    [0, scaling_factor, random_translation_y]
                ])
                
        target_shape = (target_width, target_height)
        img = self._apply_transformation(img, transformation_matrix, target_shape)
        return img

    def random_transformation(self, img):
        target_width, target_height = self.img_size
        image_height, image_width = img.shape

        scaling_factor = min(target_width / image_width, target_height / image_height)

        random_scaling_factor_x = scaling_factor * np.random.uniform(0.75, 1.05)
        random_scaling_factor_y = scaling_factor * np.random.uniform(0.75, 1.05)

            # random position around center
        center_translation_x = (target_width - image_width * random_scaling_factor_x) / 2
        center_translation_y = (target_height - image_height * random_scaling_factor_y) / 2
        freedom_x = max((target_width - random_scaling_factor_x * image_width) / 2, 0)
        freedom_y = max((target_height - random_scaling_factor_y * image_height) / 2, 0)
        random_translation_x = center_translation_x + np.random.uniform(-freedom_x, freedom_x)
        random_translation_y = center_translation_y + np.random.uniform(-freedom_y, freedom_y)

            # map image into target image
        transformation_matrix = np.float32([
                    [random_scaling_factor_x, 0, random_translation_x], 
                    [0, random_scaling_factor_y, random_translation_y]
                ])
        target_shape = self.img_size
        img = self._apply_transformation(img, transformation_matrix, target_shape)
        return img

    def _apply_transformation(self, img, transformation_matrix, target_shape):
        target_image = np.ones(target_shape[::-1]) * 255
        img = cv2.warpAffine(img, transformation_matrix, dsize=target_shape, dst=target_image, borderMode=cv2.BORDER_TRANSPARENT)
        return img

    def random_erode(self, img, probability = 0.25):
        if random.random() < probability:
            img = self.erode(img)
        return img

    def erode(self, img):
        kernel = np.ones((3, 3))
        img = cv2.erode(img, kernel)
        return img

    def random_dilate(self, img, probability = 0.25):
        if random.random() < probability:
            img = self.dilate(img)
        return img

    def dilate(self, img):
        kernel = np.ones((3, 3))
        img = cv2.dilate(img, kernel)
        return img

    def _generate_random_odd_number(self):
        return random.randint(1, 3) * 2 + 1

    def random_gaussian_blur(self, img, probability = 0.25):
        if random.random() < probability:
            random_kernel_size = (self._generate_random_odd_number(), self._generate_random_odd_number())
            img = self.gaussian_blur(img, random_kernel_size)
        return img

    def gaussian_blur(self, img, kernel_size):
        img = cv2.GaussianBlur(img, kernel_size, 0)
        return img

    def process_batch(self, batch: Batch) -> Batch:
        if self.line_mode:
            batch = self._simulate_text_line(batch)

        res_imgs = [self.process_img(img) for img in batch.imgs]
        max_text_len = res_imgs[0].shape[0] // 4
        res_gt_texts = [self._truncate_label(gt_text, max_text_len) for gt_text in batch.gt_texts]
        return Batch(res_imgs, res_gt_texts, batch.batch_size)
    

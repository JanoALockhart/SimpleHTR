import pickle
import random
from typing import List, Tuple

import cv2
import lmdb
import numpy as np
from path import Path

from dataset import AbstractDataset, Batch, Sample
from preprocessor import Preprocessor

class DataLoaderIAM:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 train_split: float = 0.95,
                 validation_split: float = 0.04,
                 fast: bool = True) -> None:
        """Loader for dataset."""

        assert data_dir.exists()

        self.samples = self.load_samples(data_dir)
        self.char_list = self._build_char_list()
        self.train_dataset, self.validation_dataset, self.test_dataset = self.split_dataset(
            train_split, 
            validation_split, 
            batch_size, 
            data_dir, 
            fast
        )
        self.corpus = [x.gt_text for x in self.samples]

    def _build_char_list(self):
        alphabet = set()
        for example in self.samples:
            unique_letters = set(list(example.gt_text))
            alphabet.union(unique_letters)
        return sorted(list(alphabet))   

    def split_dataset(self, train_split, validation_split, batch_size, data_dir, fast) -> Tuple[AbstractDataset, AbstractDataset, AbstractDataset]:
        # split into training and validation set: 95% - 4% - 1%
        dataset_size = len(self.samples)
        train_size = int(train_split * dataset_size)
        validation_size = int(validation_split * dataset_size)

        train_samples = self.samples[0:train_size]
        train_dataset = Dataset(train_samples, batch_size, data_dir, drop_remainder=True, shuffle=True, fast=fast)

        validation_samples = self.samples[train_size:train_size + validation_size]
        validation_dataset = Dataset(validation_samples, batch_size, data_dir)

        test_samples = self.samples[train_size + validation_size:]
        test_dataset = Dataset(test_samples, batch_size, data_dir)

        return train_dataset, validation_dataset, test_dataset

    def load_samples(self, data_dir) -> List[Sample]:
        ground_truths_file = open(data_dir / 'gt/words.txt')
        samples = []
        
        for line in ground_truths_file:
            line = line.strip()
            line_split = line.split(' ')

            if not self._ignore_line(line, line_split):
                image_path = self.parse_image_path(data_dir, line_split)
                ground_truth_text = ' '.join(line_split[8:]) # GT text are columns starting at 9
                samples.append(Sample(ground_truth_text, image_path))

        return samples

    def _ignore_line(self, line, line_split):
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
        first_char = line[0]
        file_name = line_split[0]
        ignore_line = not line or first_char == '#' or file_name in bad_samples_reference
        if file_name in bad_samples_reference:
            print('Ignoring known broken image:', file_name)

        return ignore_line
    
    def parse_image_path(self, data_dir, line_split):
        # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
        file_name_split = line_split[0].split('-')
        fist_subdir_name = file_name_split[0]
        second_subdir_name = f'{file_name_split[0]}-{file_name_split[1]}'
        file_name = line_split[0] + '.png'
        file_path = data_dir / 'img' / fist_subdir_name / second_subdir_name / file_name
        return file_path   
    
class Dataset(AbstractDataset):
    def __init__(self, 
                 samples:List[Sample], 
                 batch_size:int,
                 data_dir: Path,
                 drop_remainder:bool = False, 
                 shuffle:bool = False,
                 fast: bool = True):
        self.samples = samples
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.shuffle = shuffle
        self.fast = fast
        self.curr_idx = 0

        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)


    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.drop_remainder:
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller
    
    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.drop_remainder:
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def get_next(self) -> Batch:
        """Get next element."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))

    def _get_img(self, i: int) -> np.ndarray:
        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img
    
    def reset_iterator(self):
        if self.shuffle:
            random.shuffle(self.samples)
        self.curr_idx = 0
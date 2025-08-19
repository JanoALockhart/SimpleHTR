from abc import ABC, abstractmethod
import random
from typing import List, Tuple

import numpy as np

from dataset_structure import Batch, Sample
from image_loader import BaseImageLoader, ImageLoader
from preprocessor import Preprocessor

class Dataset(ABC):
    @abstractmethod
    def has_next(self) -> bool:
        pass

    @abstractmethod
    def get_next(self) -> Batch:
        pass

    @abstractmethod
    def get_iterator_info(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def map(self, image_loader: ImageLoader) -> "Dataset":
        pass

    @abstractmethod
    def map(self, preprocessor: Preprocessor) -> "Dataset":
        pass

    @abstractmethod
    def batch(self, batch_size: int, drop_remainder: bool = False) -> "Dataset":
        pass

    @abstractmethod
    def shuffle(self) -> "Dataset":
        pass

    @staticmethod
    def dataset_from_sample_list(sample_list: List[Sample]) -> "Dataset":
        return BaseDataset(sample_list)

class BaseDataset(Dataset):
    def __init__(self, samples:List[Sample]):
        self.samples = samples
        self.batch_size = 32
        self.drop_remainder = False
        self.must_shuffle = False
        self.curr_idx = 0
        self.image_loader = BaseImageLoader()
        self.preprocessor: Preprocessor = None

    def map(self, image_loader: ImageLoader):
        self.image_loader = image_loader

    def map(self, preprocessor: Preprocessor) -> Dataset:
        self.preprocessor = preprocessor
        return self

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

        imgs = [self.image_loader.load_image(self.samples[i].file_path) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        batch = Batch(imgs, gt_texts, len(imgs))
        if self.preprocessor is not None:
            batch = self.preprocessor.process_batch(batch)

        return batch
    
    def reset_iterator(self):
        if self.must_shuffle:
            random.shuffle(self.samples)
        self.curr_idx = 0

    def batch(self, batch_size: int, drop_remainder:bool = False):
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        return self
    
    def shuffle(self):
        self.must_shuffle = True
        return self

    
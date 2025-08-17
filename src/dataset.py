from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import random
from typing import List, Tuple

import cv2
import lmdb
import numpy as np
from path import Path

from preprocessor import Preprocessor

@dataclass
class Sample:
    gt_text: str
    file_path: str

@dataclass
class Batch:
    imgs: list
    gt_texts: list
    batch_size: int

class ImagePreprocessor(ABC):
    @abstractmethod
    def process_batch(self, batch: Batch) -> Batch:
        pass

class ImageLoader(ABC):
    @abstractmethod
    def get_img(self):
        pass

class AbstractDataset(ABC):
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
    def map(self, preprocessor: Preprocessor) -> "AbstractDataset":
        pass

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
        self.preprocessor: Preprocessor = None

        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

    def map(self, preprocessor: Preprocessor) -> AbstractDataset:
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

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        batch = Batch(imgs, gt_texts, len(imgs))
        if Preprocessor is not None:
            self.preprocessor.process_batch(batch)
            
        return batch

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

        
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple

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

class DatasetLoader(ABC):
    @abstractmethod
    def get_datasets(self) -> Tuple[AbstractDataset, AbstractDataset, AbstractDataset]:
        """Returns the training, validation and test datasets"""
        pass
 
        
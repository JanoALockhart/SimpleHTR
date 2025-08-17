from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple

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

class DatasetLoader(ABC):
    @abstractmethod
    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Returns the training, validation and test datasets"""
        pass
 
        
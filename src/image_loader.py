from abc import ABC, abstractmethod
import pickle

import cv2
import lmdb
from path import Path


class ImageLoader(ABC):
    @abstractmethod
    def load_image(self, path: str):
        pass

class BaseImageLoader(ImageLoader):
    def load_image(self, path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

class LMDBImageLoader(ImageLoader):
    def __init__(self, data_dir: str):
        self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

    def load_image(self, path):
        with self.env.begin() as txn:
            basename = Path(path).basename()
            data = txn.get(basename.encode("ascii"))
            img = pickle.loads(data)
        return img
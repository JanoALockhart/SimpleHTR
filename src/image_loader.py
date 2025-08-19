from abc import ABC, abstractmethod


class ImageLoader(ABC):
    @abstractmethod
    def load_image(self, path: str):
        pass
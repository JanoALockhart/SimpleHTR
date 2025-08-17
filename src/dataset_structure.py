
from dataclasses import dataclass

@dataclass
class Sample:
    gt_text: str
    file_path: str

@dataclass
class Batch:
    imgs: list
    gt_texts: list
    batch_size: int
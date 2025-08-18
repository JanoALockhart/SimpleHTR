from typing import List
import numpy as np
from dataclasses import dataclass

@dataclass
class Sample:
    gt_text: str
    file_path: str

@dataclass
class Batch:
    imgs: List[np.ndarray]
    gt_texts: List[str]
    batch_size: int
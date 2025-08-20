from abc import ABC, abstractmethod
from typing import List, Tuple

from dataset import Dataset
from dataset_structure import Sample
from image_loader import BaseImageLoader, LMDBImageLoader
from preprocessor import Preprocessor
from settings import Settings

class DatasetLoader(ABC):
    @abstractmethod
    def get_alphabet(self) -> List[str]:
        pass

    @abstractmethod
    def get_corpus(self) -> str:
        pass

    @abstractmethod
    def get_raw_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        pass

    @abstractmethod
    def get_configured_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        pass

class BaseDatasetLoader(DatasetLoader):
    samples: List[Sample]
    alphabet: List[str]
    corpus: List[str]

    def __init__(self,  data_dir: str, train_split: float = 0.95, validation_split:float = 0.04) -> None:
        """Basic loader for only one dataset from the filesystem"""
        self.data_dir = data_dir
        self.train_split = train_split
        self.validation_split = validation_split
        self.samples = self._load_samples()

        self.alphabet = self._build_alphabet()
        self.corpus = self._build_corpus()

    @abstractmethod
    def _load_samples(self):
        """Override this method according to the dataset file structure and the ground truth file format"""
        pass

    def _build_corpus(self):
        corpus_list = [x.gt_text for x in self.samples]
        return ' '.join(corpus_list)

    def _build_alphabet(self):
        alphabet = set()
        alphabet.add(' ')
        for example in self.samples:
            unique_letters = set(list(example.gt_text))
            alphabet = alphabet.union(unique_letters)
        return sorted(list(alphabet))   

    def _split_samples(self) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        dataset_size = len(self.samples)
        train_size = int(self.train_split * dataset_size)
        validation_size = int(self.validation_split * dataset_size)

        train_samples = self.samples[0:train_size]
        validation_samples = self.samples[train_size:train_size + validation_size]
        test_samples = self.samples[train_size + validation_size:]

        return train_samples, validation_samples, test_samples

    def get_raw_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        train_samples, validation_samples, test_samples = self._split_samples()

        train_set = Dataset.dataset_from_sample_list(train_samples)
        validation_set = Dataset.dataset_from_sample_list(validation_samples)
        test_set = Dataset.dataset_from_sample_list(test_samples)

        return train_set, validation_set, test_set

    
class IAMDataLoader(BaseDatasetLoader):
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """
    def __init__(self, 
                 data_dir: str, 
                 batch_size: int,
                 line_mode: bool,
                 fast: bool,
                 train_split: float = 0.95, 
                 validation_split: float = 0.04,
                 ):
        super().__init__(data_dir, train_split, validation_split)
        self.batch_size = batch_size
        self.line_mode = line_mode
        self.fast = fast

    def get_alphabet(self):
        return self.alphabet
    
    def get_corpus(self):
        # TODO Remove this writing to a file 
        with open(Settings.CORPUS_FILE_PATH, 'w') as f:
            f.write(self.corpus)

        return self.corpus
    
    def _load_samples(self) -> List[Sample]:
        data_dir = self.data_dir
        ground_truths_file = open(data_dir / 'ascii/words.txt')
        samples = []
        
        for line in ground_truths_file:
            line = line.strip()
            line_split = line.split(' ')

            if not self._ignore_line(line, line_split):
                image_path = self._parse_image_path(data_dir, line_split)
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
    
    def _parse_image_path(self, data_dir, line_split):
        # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
        file_name_split = line_split[0].split('-')
        fist_subdir_name = file_name_split[0]
        second_subdir_name = f'{file_name_split[0]}-{file_name_split[1]}'
        file_name = line_split[0] + '.png'
        file_path = data_dir / 'words' / fist_subdir_name / second_subdir_name / file_name
        return file_path   

    def get_configured_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        image_loader = BaseImageLoader()
        if self.fast:
            image_loader = LMDBImageLoader(self.data_dir)
        
        train_set, validation_set, test_set = super().get_raw_datasets()

        train_preprocessor = Preprocessor(data_augmentation=True, line_mode=self.line_mode)
        train_set.set_image_loader(image_loader).map(train_preprocessor).batch(self.batch_size, drop_remainder=True).shuffle()

        not_training_preprocessor = Preprocessor(line_mode=self.line_mode)
        validation_set.set_image_loader(image_loader).map(not_training_preprocessor).batch(self.batch_size)
        test_set.set_image_loader(image_loader).map(not_training_preprocessor).batch(self.batch_size)
        
        return train_set, validation_set, test_set
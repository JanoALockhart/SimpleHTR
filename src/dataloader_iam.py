from abc import ABC, abstractmethod
from typing import List, Tuple

from path import Path

from dataset import AbstractDataset, Dataset
from dataset_structure import Sample
from settings import Settings

class DatasetLoader(ABC):
    samples: List[Sample]
    alphabet: List[str]
    corpus: List[str]

    def __init__(self, data_dir: Path) -> None:
        """Loader for dataset."""
        assert data_dir.exists()

        self.samples = self._load_samples(data_dir)

        self.alphabet = self._build_alphabet()
        self.corpus = [x.gt_text for x in self.samples]

    def get_datasets(self, train_split: float = 0.95, validation_split: float = 0.04) -> Tuple[AbstractDataset, AbstractDataset, AbstractDataset]:
        train_samples, validation_samples, test_samples = self._split_samples(
            self.samples,
            train_split, 
            validation_split
        )

        train_set = Dataset(train_samples, drop_remainder=True, shuffle=True)
        validation_set = Dataset(validation_samples)
        test_set = Dataset(test_samples)

        return train_set, validation_set, test_set

    @abstractmethod
    def get_alphabet(self) -> List[str]:
        pass

    @abstractmethod
    def get_corpus(self) -> str:
        pass

    @abstractmethod
    def _load_samples(self, data_dir: str):
        pass

    def _build_alphabet(self):
        alphabet = set()
        alphabet.add(' ')
        for example in self.samples:
            unique_letters = set(list(example.gt_text))
            alphabet = alphabet.union(unique_letters)
        return sorted(list(alphabet))   

    def _split_samples(self, train_split, validation_split) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        # split into training and validation set: 95% - 4% - 1%
        dataset_size = len(self.samples)
        train_size = int(train_split * dataset_size)
        validation_size = int(validation_split * dataset_size)

        train_samples = self.samples[0:train_size]
        validation_samples = self.samples[train_size:train_size + validation_size]
        test_samples = self.samples[train_size + validation_size:]

        return train_samples, validation_samples, test_samples

    
class IAMDataLoader(DatasetLoader):
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def get_alphabet(self):
        return self.alphabet
    
    def get_corpus(self):
        with open(Settings.CORPUS_FILE_PATH, 'w') as f:
            f.write(' '.join(self.corpus))

        return self.corpus
    
    def _load_samples(self, data_dir) -> List[Sample]:
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

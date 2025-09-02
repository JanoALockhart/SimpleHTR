from abc import ABC, abstractmethod

import editdistance

class Metric(ABC):

    def __init__(self):
        self.info = ""

    @abstractmethod
    def update_state(self, y_true, y_prediction):
        pass

    @abstractmethod
    def result(self):
        pass

    @abstractmethod
    def reset_states(self):
        pass

    def get_info(self):
        return self.info

class CharacterErrorRate(Metric):
    def __init__(self):
        super().__init__()
        self.num_char_error = 0
        self.num_char_total = 0

    def update_state(self, y_true: str, y_prediction: str):
        char_dist = editdistance.eval(y_true, y_prediction)
        self.num_char_error += char_dist
        self.num_char_total += len(y_true)
        self.info = '[C_OK]' if char_dist == 0 else f'[C_ERR:{char_dist}]'

    def result(self):
        return self.num_char_error / self.num_char_total
    
    def reset_states(self):
        self.num_char_error = 0
        self.num_char_total = 0

class WordErrorRate(Metric):
    def __init__(self):
        super().__init__()
        self.num_word_err = 0
        self.num_word_total = 0

    def update_state(self, y_true: str, y_prediction: str):
        ground_truth_words = y_true.split(' ')
        recognized_words = y_prediction.split(' ')
        word_dist = editdistance.eval(ground_truth_words, recognized_words)
        self.num_word_err += word_dist
        self.num_word_total += len(ground_truth_words)
        self.info = '[W_OK]' if word_dist == 0 else f'[W_ERR:{word_dist}]'

    def result(self):
        return self.num_word_err / self.num_word_total
    
    def reset_states(self):
        self.num_word_err = 0
        self.num_word_total = 0

class PhraseAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.num_phrase_ok = 0
        self.num_phrase_total = 0

    def update_state(self, y_true:str, y_prediction:str):
        self.num_phrase_ok += 1 if y_true == y_prediction else 0
        self.num_phrase_total += 1
        self.info = '[P_OK]' if y_true == y_prediction else '[P_ERR]'

    def result(self):
        return self.num_phrase_ok / self.num_phrase_total
    
    def reset_states(self):
        self.num_phrase_ok = 0
        self.num_phrase_total = 0
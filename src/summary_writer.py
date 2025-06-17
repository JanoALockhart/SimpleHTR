import json

class EpochSummary:
    def __init__(self, 
                 char_error_rate: float, 
                 word_accuracies: float, 
                 average_train_loss: float,
                 time_to_train_epoch: float
                ):
        
        self.char_error_rate:float = char_error_rate
        self.word_accuracies:float = word_accuracies
        self.average_train_loss:float = average_train_loss
        self.time_to_train_epoch:float = time_to_train_epoch 


class SummaryWriter:
    def __init__(self, path):
        self.path = path
        self.summaries = []

    def append(self, epoch_summary: EpochSummary):
       self.summaries.append(epoch_summary)
       with open(self.path, 'w') as file:
           json.dump({self.summaries}, file, ident=4)


class EpochSummary:
    def __init__(self, 
                 char_error_rate: float, 
                 word_accuracies: float, 
                 average_train_loss: float,
                 time_to_train_epoch: float
                ):
        
        self.char_error_rate = char_error_rate
        self.word_accuracies = word_accuracies
        self.average_train_loss = average_train_loss
        self.time_to_train_epoch = time_to_train_epoch 

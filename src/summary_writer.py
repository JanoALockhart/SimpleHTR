import json

from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class EpochSummary:
    epoch:int
    train_cer:float
    val_cer:float
    train_wer:float
    val_wer:float
    train_phrase_acc:float
    val_phrase_acc:float
    train_loss:float
    val_loss:float
    time_to_train_epoch:float 


class SummaryWriter:
    def __init__(self, path):
        self.path = path
        self.summaries = []

    def append(self, epoch_summary: EpochSummary) -> None:
       self.summaries.append(asdict(epoch_summary))
       with open(self.path, 'w') as file:
           json.dump(self.summaries, file, indent=4)
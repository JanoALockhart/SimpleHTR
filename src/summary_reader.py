import json
import matplotlib.pyplot as plt

from summary_writer import EpochSummary

def plot_metric(title, train_metric, val_metric):
    train_line = plt.plot(train_metric)
    val_line = plt.plot(val_metric)
    plt.setp(train_line, label=f"train {title}")
    plt.setp(val_line, label=f"val {title}")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("../model/"+title+".png")
    plt.close()
    

def plot_summary():
    with open('../model/summary.json','r') as file:
        data = json.load(file)

    epochs = [EpochSummary(**epoch_summary) for epoch_summary in data]

    train_loss = [epoch_summary.train_loss for epoch_summary in epochs]
    val_loss = [epoch_summary.val_loss for epoch_summary in epochs]
    plot_metric("Loss", train_loss, val_loss)

    train_cer = [epoch_summary.train_cer for epoch_summary in epochs]
    val_cer = [epoch_summary.val_cer for epoch_summary in epochs]
    plot_metric("CER", train_cer, val_cer)
    
    train_wer = [epoch_summary.train_wer for epoch_summary in epochs]
    val_wer = [epoch_summary.val_wer for epoch_summary in epochs]
    plot_metric("WER", train_wer, val_wer)

    train_phrase_acc = [epoch_summary.train_phrase_acc for epoch_summary in epochs]
    val_phrase_acc = [epoch_summary.val_phrase_acc for epoch_summary in epochs]
    plot_metric("Phrase Accuracy", train_phrase_acc, val_phrase_acc)

    total_time_to_train = sum([epoch_summary.time_to_train_epoch for epoch_summary in epochs])
    
    saved_model_val_cer = min(val_cer)
    epoch_saved = val_cer.index(saved_model_val_cer)
    saved_model_train_cer = train_cer[epoch_saved]
    saved_model_train_wer = train_wer[epoch_saved]
    saved_model_val_wer = val_wer[epoch_saved]
    saved_model_train_phrase_acc = train_phrase_acc[epoch_saved]
    saved_model_val_phrase_acc = val_phrase_acc[epoch_saved]

    with open("../model/description.txt", 'w') as file:
        file.write(f"Total Epochs: {len(epochs)} \n")
        file.write(f"Time to train: {total_time_to_train / 3600: .2f}hs \n")
        file.write(f"Train CER: {saved_model_train_cer*100: .2f}% \n")
        file.write(f"Validation CER: {saved_model_val_cer*100: .2f}% \n")
        file.write(f"Train WER: {saved_model_train_wer*100: .2f}% \n")
        file.write(f"Validation WER: {saved_model_val_wer*100: .2f}% \n")
        file.write(f"Train Phrase Accuracy: {saved_model_train_phrase_acc*100: .2f}% \n")
        file.write(f"Validation Phrase Accuracy: {saved_model_val_phrase_acc*100: .2f}% \n")

if __name__ == '__main__':
    plot_summary()
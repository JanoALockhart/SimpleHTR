import json
import matplotlib.pyplot as plt

from summary_writer import EpochSummary

def plot_metric(title, metric):
    line = plt.plot(metric)
    plt.setp(line, label=title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("../model/"+title+".png")
    #plt.show()
    plt.close()
    

def plot_summary():
    with open('../model/summary.json','r') as file:
        data = json.load(file)

    epochs = [EpochSummary(**epoch_summary) for epoch_summary in data]

    loss = [epoch_summary.average_train_loss for epoch_summary in epochs]
    character_error_rate = [epoch_summary.char_error_rate for epoch_summary in epochs]
    word_accuracy = [epoch_summary.phrase_accuracies for epoch_summary in epochs]
    
    plot_metric("Loss", loss)
    plot_metric("CER", character_error_rate)
    plot_metric("Phrase Accuracy", word_accuracy)

    total_time_to_train = sum([epoch_summary.time_to_train_epoch for epoch_summary in epochs])
    saved_model_cer = min(character_error_rate)
    saved_model_word_accuracy = word_accuracy[character_error_rate.index(saved_model_cer)]

    with open("../model/description.txt", 'w') as file:
        file.write(f"Total Epochs: {len(epochs)} \n")
        file.write(f"Time to train (hours): {total_time_to_train / 3600: .2f} \n")
        file.write(f"CER: {saved_model_cer*100: .2f}% \n")
        file.write(f"Phrase Accuracy: {saved_model_word_accuracy*100: .2f}% \n")

if __name__ == '__main__':
    plot_summary()
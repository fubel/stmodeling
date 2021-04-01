import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy(history,model_details):
    plt.plot(history.accuracy)
    plt.plot(history.val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.savefig(f'results/plots/{model_details['backbone']}_{model_details['transformer_arch']}_lr_{model_details['lr']:.5f}_bs_{model_details['batch_size']}_accuracy.png',bbox_inches='tight')

def plot_loss(history,model_details):
    plt.plot(history.loss)
    plt.plot(history.val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.savefig(f'results/plots/{model_details['backbone']}_{model_details['transformer_arch']}_lr_{model_details['lr']:.5f}_bs_{model_details['batch_size']}_loss.png',bbox_inches='tight')

def plot_statistics(history,model_details):
    plot_accuracy(history,model_details)
    plot_loss(history,model_details)
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy(history,model_details):
    plt.figure(0)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim([0, 40])
    plt.ylim([0, 100])
    plt.legend(['train','validation'], loc='upper left')
    plt.savefig('results/plots/{}_{}_lr_{:.5f}_bs_{}_accuracy.png'.format(model_details['backbone'],model_details['transformer_arch'],model_details['lr'],model_details['batch_size']),bbox_inches='tight')

def plot_loss(history,model_details):
    plt.figure(1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.savefig('results/plots/{}_{}_lr_{:.5f}_bs_{}_loss.png'.format(model_details['backbone'],model_details['transformer_arch'],model_details['lr'],model_details['batch_size']),bbox_inches='tight')

def plot_statistics(history,model_details):
    plot_accuracy(history,model_details)
    plot_loss(history,model_details)
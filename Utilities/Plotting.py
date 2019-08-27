from matplotlib import pyplot as plt
from datetime import datetime


def plot_accuracy_curves(results):
    fig = plt.figure(figsize=[8, 6])
    plt.plot(results['acc'], 'r', linewidth=3.0)
    plt.plot(results['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Acc', 'Validation Acc'], fontsize=18, loc='lower right')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Acc', fontsize=16)
    plt.title('Acc Curves', fontsize=16)
    fig.savefig(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' +
                str(datetime.now().second) + '_' + 'acc.png')


def plot_loss_curves(results):
    fig = plt.figure(figsize=[8, 6])
    plt.plot(results['categorical_crossentropy'], 'r', linewidth=3.0)
    plt.plot(results['val_categorical_crossentropy'], 'b', linewidth=3.0)
    plt.legend(['Training Categorical Crossentropy', 'Validation Categorical Cossentropy'], fontsize=18,
               loc='upper right')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Categorical Crossentropy', fontsize=16)
    plt.title('Categorical Crossentropy Curves', fontsize=16)
    fig.savefig(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' +
                str(datetime.now().second) + '_' + 'categorical_crossentropy.png')

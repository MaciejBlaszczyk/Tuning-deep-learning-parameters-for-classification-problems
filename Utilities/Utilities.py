import numpy as np
import os
import pandas as pd
from keras import backend as K
from Enums.ClassifierType import ClassifierType
from Enums.ModelTuningType import ModelTuningType

MLP_header = 'Dataset\tArchitecture\tType\tNeurons\tHidden Layers\tAverage Time\tBal Acc'
CNN_header = 'Dataset\tArchitecture\tType\tHidden Layers\tFilters\tFilter Size\tAverage Time\tBal Acc'
CAPS_header = 'Dataset\tArchitecture\tType\tFilters\tFilter Size\tChannels\tCapsule Dimension\tAverage Time\tBal Acc'
LSTM_header = 'Dataset\tArchitecture\tType\tHidden Layers\tCells\tEmbedded Vector Length\tAverage Time\tBal Acc'


def balanced_accuracy(y_true, y_pred):
    """Default scoring function: balanced accuracy.
    Balanced accuracy computes each class' accuracy on a per-class basis using a
    one-vs-rest encoding, then computes an unweighted average of the class accuracies.
    Parameters
    ----------
    y_true: numpy.ndarray {n_samples}
        True class labels
    y_pred: numpy.ndarray {n_samples}
        Predicted class labels by the estimator
    Returns
    -------
    fitness: float
        Returns a float value indicating the individual's balanced accuracy
        0.5 is as good as chance, and 1.0 is perfect predictive accuracy
    """
    all_classes = list(set(np.append(y_true, y_pred)))
    all_class_accuracies = []
    for this_class in all_classes:
        this_class_sensitivity = 0.
        this_class_specificity = 0.
        if sum(y_true == this_class) != 0:
            this_class_sensitivity = \
                float(sum((y_pred == this_class) & (y_true == this_class))) /\
                float(sum((y_true == this_class)))

            this_class_specificity = \
                float(sum((y_pred != this_class) & (y_true != this_class))) /\
                float(sum((y_true != this_class)))

        this_class_accuracy = (this_class_sensitivity + this_class_specificity) / 2.
        all_class_accuracies.append(this_class_accuracy)

    return np.mean(all_class_accuracies)


def write_to_file(path, *args):
    with open(path, 'a') as file:
        if os.stat(path).st_size == 0:
            if 'MLP' in args:
                file.write(MLP_header)
            if 'CNN' in args:
                file.write(CNN_header)
            if 'CAPS' in args:
                file.write(CAPS_header)
            if 'LSTM' in args:
                file.write(LSTM_header)
        text = str()
        for item in args:
            text += str(item) + '\t'
        file.write('\n' + text)


def get_best_hyperparams(type, path):
    with open(path, 'a') as file:
        df = pd.read_csv(path, sep='\t', index_col=False)
        df_filt = (df.loc[df['Type'] == ModelTuningType.GRID_SEARCH.name])
        best_params_index = df_filt['Bal Acc'].idxmax()
        if type == ClassifierType.MLP:
            return {'hidden_layers': [df['Hidden Layers'][best_params_index]],
                    'neurons': [df['Neurons'][best_params_index]],
                    'activation': 'softmax',
                    'optimizer': 'Nadam'}
        if type == ClassifierType.CNN:
            return {'hidden_layers': [int(df['Hidden Layers'][best_params_index])],
                    'filters': [int(df['Filters'][best_params_index])],
                    'filter_size': [int(df['Filter Size'][best_params_index])],
                    'activation': 'softmax',
                    'optimizer': 'Nadam'}
        if type == ClassifierType.CAPS:
            return {'filters': [int(df['Filters'][best_params_index])],
                    'filter_size': [int(df['Filter Size'][best_params_index])],
                    'channels': [int(df['Channels'][best_params_index])],
                    'caps_dim': [int(df['Capsule Dimension'][best_params_index])],
                    'activation': 'softmax',
                    'optimizer': 'Nadam'}
        if type == ClassifierType.LSTM:
            return {'hidden_layers': [int(df['Hidden Layers'][best_params_index])],
                    'cells': [int(df['Cells'][best_params_index])],
                    'embedded_vec_len': [int(df['Embedded Vector Length'][best_params_index])],
                    'activation': 'softmax',
                    'optimizer': 'Nadam'}


def count_words(dataset):
    unique_set = set()
    for element in dataset:
        for word in element:
            if word not in unique_set:
                unique_set.add(word)
    return unique_set


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))
import numpy as np
from Utilities.Utilities import balanced_accuracy
from Enums.ClassifierType import ClassifierType


class Classifier:
    def __init__(self, data_handler, epochs, batch_size, eval_type):
        self.data_handler = data_handler
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_type = eval_type

    def calculate_bal_acc(self, y_predicted, y_test):
        y_predicted[np.arange(len(y_predicted)), y_predicted.argmax(1)] = 1

        inversed_y_predicted = self.data_handler.label_binarizer.inverse_transform(y_predicted)
        inversed_y_test = self.data_handler.label_binarizer.inverse_transform(y_test)
        return balanced_accuracy(inversed_y_test, inversed_y_predicted)

    @staticmethod
    def get_parameters(classifier_type, args):
        if classifier_type == ClassifierType.MLP:
            return {'hidden_layers': args.MLP_hidden_layers,
                    'neurons': args.MLP_neurons,
                    'activation': args.activation,
                    'optimizer': args.optimizer}
        elif classifier_type == ClassifierType.CNN:
            return {'hidden_layers': args.CNN_hidden_layers,
                    'filters': args.CNN_filters,
                    'filter_size': args.CNN_filter_size,
                    'activation': args.activation,
                    'optimizer': args.optimizer}
        elif classifier_type == ClassifierType.LSTM:
            return {'hidden_layers': args.LSTM_hidden_layers,
                    'cells': args.LSTM_cells,
                    'embedded_vec_len': args.LSTM_embedded_vec_len,
                    'activation': args.activation,
                    'optimizer': args.optimizer}
        elif classifier_type == ClassifierType.CAPS:
            return {'filters': args.CAPS_filters,
                    'filter_size': args.CAPS_filter_size,
                    'channels' : args.CAPS_channels,
                    'caps_dim' : args.CAPS_caps_dim,
                    'activation': args.activation,
                    'optimizer': args.optimizer}
import gc
import os
import sys
import time

import numpy as np
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Dense
from keras.models import Sequential

from Classifiers.Classifier import Classifier
from Enums.ClassifierType import ClassifierType
from Enums.ModelTuningType import ModelTuningType
from Utilities.Utilities import write_to_file


class MLPClassifier(Classifier):
    """hyperparameters: optimizer, activation, hidden_layers, neurons"""

    def __init__(self, data_handler, batch_size, epochs, eval_type, hyperparams):
        super().__init__(data_handler, epochs, batch_size, eval_type)
        self.optimizer = hyperparams['optimizer']
        self.activation = hyperparams['activation']
        self.hidden_layers = hyperparams['hidden_layers']
        self.neurons = hyperparams['neurons']
        self.results_path = os.path.dirname(os.path.abspath(__file__)) + '/MLP_' + data_handler.dataset_name[
                                                                                   :-7] + '_Results.tsv'

    def create_model(self, hidden_layers, neurons):
        model = Sequential()
        model.add(Dense(self.data_handler.features, input_dim=self.data_handler.features))
        model.add(BatchNormalization())
        for _ in range(hidden_layers):
            model.add(Dense(neurons, activation=self.activation))
            model.add(BatchNormalization())
        model.add(Dense(self.data_handler.classes, activation='softmax'))
        return model

    def perform_classification(self):
        repeats = 4 if self.eval_type == ModelTuningType.EVALUATE_BEST else 1
        for hidden_layers in self.hidden_layers:
            for neurons in self.neurons:
                bal_accuracies = list()
                runtimes = list()
                for _ in range(repeats):
                    for train_index, test_index in self.data_handler.skf.split(self.data_handler.X_scaled,
                                                                               self.data_handler.Y):
                        y_train = self.data_handler.y_one_hot_encoded[train_index]
                        y_test = self.data_handler.y_one_hot_encoded[test_index]
                        X_test = self.data_handler.X_scaled[test_index]
                        X_train = self.data_handler.X_scaled[train_index]

                        model = self.create_model(hidden_layers, neurons)
                        model.compile(optimizer=self.optimizer,
                                      loss='categorical_crossentropy',
                                      metrics=[metrics.categorical_crossentropy, 'accuracy'])

                        t0 = time.time()
                        model.fit(X_train,
                                  y_train,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  verbose=2,
                                  class_weight=self.data_handler.class_weights,
                                  validation_data=(X_test, y_test),
                                  callbacks=[EarlyStopping(min_delta=0.001, patience=50)])
                        runtime = time.time() - t0

                        Y_predicted = model.predict(X_test)
                        bal_acc = self.calculate_bal_acc(Y_predicted, y_test)

                        bal_accuracies.append(bal_acc)
                        runtimes.append(runtime)

                sys.stdout.flush()
                gc.collect()
                K.clear_session()
                write_to_file(self.results_path,
                              self.data_handler.dataset_name,
                              ClassifierType.MLP.name,
                              self.eval_type.name,
                              neurons,
                              hidden_layers,
                              str('%.3f' % np.mean(runtimes)),
                              str('%.3f' % np.mean(bal_accuracies)))

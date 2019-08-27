import gc
import os
import sys
import time

import numpy as np
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization
from keras.models import Sequential

from Classifiers.Classifier import Classifier
from Enums.ClassifierType import ClassifierType
from Enums.ModelTuningType import ModelTuningType
from Utilities.Utilities import write_to_file


class CNNClassifier(Classifier):
    """Hyperparameters: optimizer, activation, hidden_layers, filters, filter_size"""

    def __init__(self, data_handler, batch_size, epochs, eval_type, hyperparams):
        super().__init__(data_handler, epochs, batch_size, eval_type)
        self.optimizer = hyperparams['optimizer']
        self.activation = hyperparams['activation']
        self.hidden_layers = hyperparams['hidden_layers']
        self.filters = hyperparams['filters']
        self.filter_size = hyperparams['filter_size']
        self.results_path = os.path.dirname(os.path.abspath(__file__)) + '/CNN_' + data_handler.dataset_name[
                                                                                   :-7] + '_Results.tsv'

    def create_model(self, hidden_layers, filters, filter_size):
        model = Sequential()
        model.add(Conv1D(filters, filter_size, padding='same', activation='relu',
                         input_shape=(self.data_handler.features, 1)))
        model.add(BatchNormalization())
        for _ in range(hidden_layers):
            model.add(Conv1D(filters, filter_size, padding='same', activation='relu'))
            model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(self.data_handler.classes, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.data_handler.classes, activation='softmax'))
        return model

    def perform_classification(self):
        repeats = 4 if self.eval_type == ModelTuningType.EVALUATE_BEST else 1
        for hidden_layers in self.hidden_layers:
            for filter_size in self.filter_size:
                for filters in self.filters:
                    bal_accuracies = list()
                    runtimes = list()
                    for _ in range(repeats):
                        for train_index, test_index in self.data_handler.skf.split(self.data_handler.X_scaled,
                                                                                   self.data_handler.Y):
                            y_train = self.data_handler.y_one_hot_encoded[train_index]
                            y_test = self.data_handler.y_one_hot_encoded[test_index]
                            X_test = self.data_handler.X_scaled[test_index]
                            X_train = self.data_handler.X_scaled[train_index]

                            model = self.create_model(hidden_layers, filters, filter_size)
                            model.compile(optimizer=self.optimizer,
                                          loss='categorical_crossentropy',
                                          metrics=[metrics.categorical_crossentropy, 'accuracy'])

                            X_test = np.expand_dims(X_test, axis=2)
                            X_train = np.expand_dims(X_train, axis=2)

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
                                  ClassifierType.CNN.name,
                                  self.eval_type.name,
                                  str(hidden_layers),
                                  str(filters),
                                  str(filter_size),
                                  str('%.3f' % np.mean(runtimes)),
                                  str('%.3f' % np.mean(bal_accuracies)))

import gc
import os
import sys
import time

import numpy as np
from keras import backend as K
from keras import metrics, layers, models
from keras.callbacks import EarlyStopping

from Classifiers.Classifier import Classifier
from Classifiers.capsulelayers import CapsuleLayer, PrimaryCap, Length
from Enums.ClassifierType import ClassifierType
from Enums.ModelTuningType import ModelTuningType
from Utilities.Utilities import margin_loss, write_to_file


class CAPSClassifier(Classifier):
    def __init__(self, data_handler, batch_size, epochs, eval_type, hyperparams):
        super().__init__(data_handler, epochs, batch_size, eval_type)
        self.optimizer = hyperparams['optimizer']
        self.activation = hyperparams['activation']
        self.filters = hyperparams['filters']
        self.filter_size = hyperparams['filter_size']
        self.channels = hyperparams['channels']
        self.caps_dims = hyperparams['caps_dim']
        self.routings = 3
        self.results_path = os.path.dirname(os.path.abspath(__file__)) + '/CAPS_' + data_handler.dataset_name[
                                                                                    :-7] + '_Results.tsv'

    def create_model(self, filter_size, filters, caps_dim, channel):
        x = layers.Input(shape=np.expand_dims(self.data_handler.X_scaled, axis=2).shape[1:])

        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv1D(filters=filters, kernel_size=filter_size, strides=1, padding='same',
                              activation='relu', name='conv1')(x)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap(conv1, dim_capsule=caps_dim, n_channels=channel, kernel_size=filter_size, strides=1,
                                 padding='same')

        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=self.data_handler.classes, dim_capsule=caps_dim, routings=self.routings,
                                 name='digitcaps')(primarycaps)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        out_caps = Length(name='capsnet')(digitcaps)

        train_model = models.Model(x, out_caps)
        return train_model

    def perform_classification(self):
        repeats = 4 if self.eval_type == ModelTuningType.EVALUATE_BEST else 1
        for filter_size in self.filter_size:
            for filters in self.filters:
                for caps_dim in self.caps_dims:
                    for channel in self.channels:
                        bal_accuracies = list()
                        runtimes = list()
                        for _ in range(repeats):
                            for train_index, test_index in self.data_handler.skf.split(self.data_handler.X_scaled,
                                                                                       self.data_handler.Y):
                                y_train = self.data_handler.y_one_hot_encoded[train_index]
                                y_test = self.data_handler.y_one_hot_encoded[test_index]
                                X_test = self.data_handler.X_scaled[test_index]
                                X_train = self.data_handler.X_scaled[train_index]

                                model = self.create_model(filter_size, filters, caps_dim, channel)
                                model.compile(optimizer=self.optimizer,
                                              loss=margin_loss,
                                              metrics=[metrics.categorical_crossentropy, 'accuracy'])

                                # TODO check if it doesnt break here
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
                                      ClassifierType.CAPS.name,
                                      self.eval_type.name,
                                      str(filters),
                                      str(filter_size),
                                      str(channel),
                                      str(caps_dim),
                                      str('%.3f' % np.mean(runtimes)),
                                      str('%.3f' % np.mean(bal_accuracies)))

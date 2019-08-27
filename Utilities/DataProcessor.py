import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

from keras.utils import to_categorical


class DataProcessor:
    def __init__(self, path, dataset_name, comp='gzip', sep='\t'):
        self.scaler = StandardScaler()
        self.label_binarizer = LabelBinarizer()
        self.dataset_name = dataset_name

        dataframe = pd.read_csv(path + '/' + dataset_name, compression=comp, sep=sep)
        self.X = dataframe.drop('target', axis=1).values.astype(np.float32)
        self.Y = dataframe['target'].values.astype(np.int32)

        self.features = len(self.X[0])

    def scale_features(self):
        self.X_scaled = self.scaler.fit_transform(self.X)

    def binarize_labels(self):
        self.label_binarizer.fit(self.Y)
        self.classes = len(self.label_binarizer.classes_)
        self.y_one_hot_encoded = self.label_binarizer.transform(self.Y)
        if self.classes == 2:
            self.y_one_hot_encoded = to_categorical(self.y_one_hot_encoded)

    def compute_class_weights(self):
        le = LabelEncoder()
        y_integers = le.fit_transform(list(self.Y))
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        self.class_weights = dict(zip(le.transform(list(le.classes_)), class_weights))

    def split_into_folds(self, n_splits=5, shuffle=True):
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)

    @staticmethod
    def load_parameters_for_best_settings(path, *args):
        df = pd.read_csv(path, sep='\t')
        df = df.sort_values('Bal Acc')
        parameters = dict()
        for arg in args:
            parameters[arg] = df[arg].values[0]
        return parameters

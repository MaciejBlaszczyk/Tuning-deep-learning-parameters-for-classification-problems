from Utilities.DataProcessor import DataProcessor
from Utilities.Utilities import get_best_hyperparams
from Classifiers.Classifier import Classifier
from Classifiers.MLPClassifier import MLPClassifier
from Classifiers.LSTMClassifier import LSTMClassifier
from Classifiers.CNNClassifier import CNNClassifier
from Classifiers.CAPSClassifier import CAPSClassifier
from Enums.ModelTuningType import ModelTuningType
from Enums.ClassifierType import ClassifierType
import os


def get_classifier(type, data_handler, batch_size, epochs, eval_type, hyperparams):
    if type == ClassifierType.MLP:
        return MLPClassifier(data_handler, batch_size, epochs, eval_type, hyperparams)
    elif type == ClassifierType.CNN:
        return CNNClassifier(data_handler, batch_size, epochs, eval_type, hyperparams)
    elif type == ClassifierType.LSTM:
        return LSTMClassifier(data_handler, batch_size, epochs, eval_type, hyperparams)
    elif type == ClassifierType.CAPS:
        return CAPSClassifier(data_handler, batch_size, epochs, eval_type, hyperparams)
    else:
        print("wrong classifier type provided")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLP")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--datasets', default='\Datasets\penn-ml-benchmarks\datasets\classification\\confidence',
                        type=str)
    parser.add_argument('--grid_search', default=True, type=bool)
    parser.add_argument('--evaluate_best', default=True, type=bool)
    parser.add_argument('--optimizer', default='Nadam', type=str)
    parser.add_argument('--activation', default='softmax', type=str)

    parser.add_argument('--MLP_hidden_layers', default=[1, 2, 3, 4], type=list)
    parser.add_argument('--MLP_neurons', default=[5, 10, 20, 50, 100], type=list)

    parser.add_argument('--CNN_hidden_layers', default=[1, 2, 3], type=list)
    parser.add_argument('--CNN_filters', default=[5, 10, 20, 50], type=list)
    parser.add_argument('--CNN_filter_size', default=[2], type=list)

    parser.add_argument('--CAPS_filter_size', default=[2, 4, 6], type=list)
    parser.add_argument('--CAPS_filters', default=[5, 10, 20,50], type=list)
    parser.add_argument('--CAPS_caps_dim', default=[4, 10, 15], type=list)
    parser.add_argument('--CAPS_channels', default=[4, 10, 15], type=list)

    parser.add_argument('--LSTM_hidden_layers', default=[1, 2, 3], type=list)
    parser.add_argument('--LSTM_cells', default=[5, 10, 20, 50], type=list)
    parser.add_argument('--LSTM_embedded_vec_len', default=[4, 8, 12, 16], type=list)

    parser.add_argument('--type', default='CAPS', type=str)
    args = parser.parse_args()
    print("Type " + str(args.type))
    print("Dataset: " + str(args.datasets))

    datasets_path = os.path.dirname(os.path.abspath(__file__)) + '/' + args.datasets
    file_name = ''
    for file in os.listdir(datasets_path):
        if file.endswith('tsv.gz'):
            file_name = file

    data_processor = DataProcessor(datasets_path, file_name)
    data_processor.scale_features()
    data_processor.binarize_labels()
    data_processor.split_into_folds()
    data_processor.compute_class_weights()

    hyperparams = Classifier.get_parameters(ClassifierType[args.type], args)

    classifier = get_classifier(ClassifierType[args.type],
                                data_processor,
                                args.batch_size,
                                args.epochs,
                                ModelTuningType.GRID_SEARCH,
                                hyperparams)
    classifier.perform_classification()

    print('\n' + args.datasets[0] + ' training completed.')

    best_hyperparams = get_best_hyperparams(ClassifierType[args.type], classifier.results_path)
    g = 1
    classifier = get_classifier(ClassifierType[args.type],
                                data_processor,
                                args.batch_size,
                                args.epochs,
                                ModelTuningType.EVALUATE_BEST,
                                best_hyperparams)
    classifier.perform_classification()

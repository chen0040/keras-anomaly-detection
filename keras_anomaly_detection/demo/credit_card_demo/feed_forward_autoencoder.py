import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras_anomaly_detection.library.feedforward import FeedForwardAutoEncoder
from keras_anomaly_detection.demo.credit_card_demo.unzip_utils import unzip
from keras_anomaly_detection.library.plot_utils import plot_confusion_matrix, plot_training_history, visualize_anomaly
from keras_anomaly_detection.library.report_utils import report_evaluation_metrics
import numpy as np

DO_TRAINING = True


def preprocess_data(csv_data):
    credit_card_data = csv_data.drop(labels=['Class', 'Time'], axis=1)
    credit_card_data['Amount'] = StandardScaler().fit_transform(credit_card_data['Amount'].values.reshape(-1, 1))
    # print(credit_card_data.head())
    credit_card_np_data = credit_card_data.as_matrix()
    y_true = csv_data['Class'].as_matrix()
    return credit_card_np_data, y_true


def main():
    seed = 42
    np.random.seed(seed)

    data_dir_path = './data'
    model_dir_path = './models'

    # ecg data in which each row is a temporal sequence data of continuous values
    unzip(data_dir_path + '/creditcardfraud.zip', data_dir_path)
    csv_data = pd.read_csv(data_dir_path + '/creditcard.csv')
    estimated_negative_sample_ratio = 1 - csv_data['Class'].sum() / csv_data['Class'].count()
    print(estimated_negative_sample_ratio)
    X, Y = preprocess_data(csv_data)
    print(X.shape)

    ae = FeedForwardAutoEncoder()

    training_history_file_path = model_dir_path + '/' + FeedForwardAutoEncoder.model_name + '-history.npy'
    # fit the data and save model into model_dir_path
    epochs = 100
    history = None
    if DO_TRAINING:
        history = ae.fit(X, model_dir_path=model_dir_path,
                         estimated_negative_sample_ratio=estimated_negative_sample_ratio,
                         nb_epoch=epochs,
                         random_state=seed)
        np.save(training_history_file_path, history)
    else:
        history = np.load(training_history_file_path).item()

    # load back the model saved in model_dir_path detect anomaly
    y_pred = []
    ae.load_model(model_dir_path)
    X, Y = preprocess_data(csv_data)
    _, Xtest, _, Ytest = train_test_split(X, Y, test_size=0.2, random_state=seed)
    reconstruction_error = []
    anomaly_information = ae.anomaly(Xtest, 2.9)
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        predicted_label = 1 if is_anomaly else 0
        # print('# ' + str(idx) + ', actual: ' + str(actual_label) + ', predicted: ' + str(predicted_label))
        y_pred.append(predicted_label)
        reconstruction_error.append(dist)

    report_evaluation_metrics(Ytest, y_pred)
    plot_training_history(history)
    visualize_anomaly(Ytest, reconstruction_error, ae.threshold)
    plot_confusion_matrix(Ytest, y_pred)


if __name__ == '__main__':
    main()

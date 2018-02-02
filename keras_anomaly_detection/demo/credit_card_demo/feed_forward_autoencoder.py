import pandas as pd
from sklearn.preprocessing import StandardScaler

from keras_anomaly_detection.library.feedforward import FeedForwardAutoEncoder
from keras_anomaly_detection.demo.credit_card_demo.unzip_utils import unzip
from keras_anomaly_detection.library.plot_utils import plot_confusion_matrix, plot_training_history, visualize_anomaly
from keras_anomaly_detection.library.report_utils import report_evaluation_metrics
import numpy as np

DO_TRAINING = False


def preprocss_data(csv_data):
    creditcard_data = csv_data.drop(labels=['Class', 'Time'], axis=1)
    creditcard_data['Amount'] = StandardScaler().fit_transform(creditcard_data['Amount'].values.reshape(-1, 1))
    print(creditcard_data.head())
    creditcard_np_data = creditcard_data.as_matrix()
    return creditcard_np_data


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
    credit_card_np_data = preprocss_data(csv_data)
    print(credit_card_np_data.shape)

    ae = FeedForwardAutoEncoder()

    training_history_file_path = model_dir_path + '/' + FeedForwardAutoEncoder.model_name + '-history.npy'
    # fit the data and save model into model_dir_path
    epochs = 100
    history = None
    if DO_TRAINING:
        history = ae.fit(credit_card_np_data, model_dir_path=model_dir_path,
                         estimated_negative_sample_ratio=estimated_negative_sample_ratio,
                         nb_epoch=epochs,
                         random_state=seed)
        np.save(training_history_file_path, history)
    else:
        history = np.load(training_history_file_path).item()

    # load back the model saved in model_dir_path detect anomaly
    y_true = []
    y_pred = []
    ae.load_model(model_dir_path)
    test_data = preprocss_data(csv_data)
    mse = ae.predict(test_data)
    anomaly_information = ae.anomaly(test_data)
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        actual_label = csv_data['Class'][idx]
        predicted_label = 1 if is_anomaly else 0
        # print('# ' + str(idx) + ', actual: ' + str(actual_label) + ', predicted: ' + str(predicted_label))
        y_true.append(actual_label)
        y_pred.append(predicted_label)

    report_evaluation_metrics(y_true, y_pred)
    plot_training_history(history)
    visualize_anomaly(y_true, mse)
    plot_confusion_matrix(y_true, y_pred)


if __name__ == '__main__':
    main()

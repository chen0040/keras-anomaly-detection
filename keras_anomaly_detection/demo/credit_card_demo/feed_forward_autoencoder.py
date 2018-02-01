import pandas as pd
from sklearn.preprocessing import StandardScaler

from keras_anomaly_detection.library.feedforward import FeedForwardAutoEncoder
from keras_anomaly_detection.demo.credit_card_demo.unzip_utils import unzip
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score

def main():
    data_dir_path = './data'
    model_dir_path = './models'

    # ecg data in which each row is a temporal sequence data of continuous values
    unzip(data_dir_path + '/creditcardfraud.zip', data_dir_path)
    csv_data = pd.read_csv(data_dir_path + '/creditcard.csv')
    estimated_negative_sample_ratio = 1 - csv_data['Class'].sum() / csv_data['Class'].count()
    print(estimated_negative_sample_ratio)
    creditcard_data = csv_data.drop(labels=['Class', 'Time'], axis=1)
    creditcard_data['Amount'] = StandardScaler().fit_transform(creditcard_data['Amount'].values.reshape(-1, 1))
    print(creditcard_data.head())
    creditcard_np_data = creditcard_data.as_matrix()

    print(creditcard_np_data.shape)

    ae = FeedForwardAutoEncoder()

    # fit the data and save model into model_dir_path
    epochs = 30
    ae.fit(creditcard_np_data, model_dir_path=model_dir_path, estimated_negative_sample_ratio=estimated_negative_sample_ratio, nb_epoch=epochs)

    # load back the model saved in model_dir_path detect anomaly
    y_true = []
    y_pred = []
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(creditcard_np_data)
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        actual_label = csv_data['Class'][idx]
        predicted_label = 1 if is_anomaly else 0
        # print('# ' + str(idx) + 'actual: ' + str(actual_label) + ', predicted: ' + str(predicted_label))
        y_true.append(actual_label)
        y_pred.append(predicted_label)

    average_precision = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    recall = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], pos_label=1)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    print('Precision: {0:0.2f}', precision)
    print('Recall: {0:0.2f}', recall)
    print('F1: {0:0.2f}', f1)


if __name__ == '__main__':
    main()

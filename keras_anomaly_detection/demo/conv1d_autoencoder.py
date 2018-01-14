import pandas as pd

from keras_anomaly_detection.library.convolutional import Conv1DAutoEncoder


def main():
    data_dir_path = '../training/data'
    model_dir_path = '../training/models'
    ecg_data = pd.read_csv(data_dir_path + '/ecg_discord_test.csv', header=None)
    print(ecg_data.head())
    ecg_np_data = ecg_data.as_matrix()
    scaler = MinMaxScaler()
    timeseries = scaler.fit_transform(ecg_np_data)

    print(ecg_np_data.shape)

    ae = Conv1DAutoEncoder()
    ae.fit(ecg_np_data[:20, :], model_dir_path=model_dir_path)
    ae.load_model(model_dir_path)
    ae.anomaly(ecg_np_data[:23, :])



if __name__ == '__main__':
    main()

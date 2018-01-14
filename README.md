# keras-anomaly-detection

Anomaly detection implemented in Keras

The source codes of the recurrent and convolutional networks auto-encoders for anomaly detection can be found in
[keras_anomaly_detection/library/convolutional.py](keras_anomaly_detection/library/convolutional.py) and
[keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py) 

The the anomaly detection is implemented using auto-encoder with convolutional and recurrent networks and can be applied
to:

* timeseries data to detect timeseries time windows that have anomaly pattern
    * LstmAutoEncoder in [keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py)
    * Conv1DAutoEncoder in [keras_anomaly_detection/library/convolutional.py](keras_anomaly_detection/library/convolutional.py)
    * CnnLstmAutoEncoder in [keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py)
    * BidirectionalLstmAutoEncoder in [keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py)
* structured data (i.e., tabular data) to detect anomaly in data records
    * Conv1DAutoEncoder in [keras_anomaly_detection/library/convolutional.py](keras_anomaly_detection/library/convolutional.py)

# Usage

The sample codes can be found in the [keras_anomaly_detection/demo](keras_anomaly_detection/demo).

The following sample codes show how to fit and detect anomaly using Conv1DAutoEncoder:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras_anomaly_detection.library.convolutional import Conv1DAutoEncoder


def main():
    data_dir_path = '../training/data'
    model_dir_path = '../training/models'

    # ecg data in which each row is a temporal sequence data of continuous values
    ecg_data = pd.read_csv(data_dir_path + '/ecg_discord_test.csv', header=None)
    print(ecg_data.head())
    ecg_np_data = ecg_data.as_matrix()
    scaler = MinMaxScaler()
    ecg_np_data = scaler.fit_transform(ecg_np_data)

    print(ecg_np_data.shape)

    ae = Conv1DAutoEncoder()

    # fit the data and save model into model_dir_path
    ae.fit(ecg_np_data[:23, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(ecg_np_data[:23, :])
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')


if __name__ == '__main__':
    main()
```

The following sample codes show how to fit and detect anomaly using LstmAutoEncoder:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras_anomaly_detection.library.recurrent import LstmAutoEncoder


def main():
    data_dir_path = '../training/data'
    model_dir_path = '../training/models'
    ecg_data = pd.read_csv(data_dir_path + '/ecg_discord_test.csv', header=None)
    print(ecg_data.head())
    ecg_np_data = ecg_data.as_matrix()
    scaler = MinMaxScaler()
    ecg_np_data = scaler.fit_transform(ecg_np_data)
    print(ecg_np_data.shape)

    ae = LstmAutoEncoder()

    # fit the data and save model into model_dir_path
    ae.fit(ecg_np_data[:23, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(ecg_np_data[:23, :])
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')


if __name__ == '__main__':
    main()
```

# Note

There is also an autoencoder from H2O for timeseries anomaly detection in 
[keras_anomaly_detection/demo/h2o_ecg_pulse_detection.py](keras_anomaly_detection/demo/h2o_ecg_pulse_detection.py)





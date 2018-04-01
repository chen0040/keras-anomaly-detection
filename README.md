# keras-anomaly-detection

Anomaly detection implemented in Keras

The source codes of the recurrent, convolutional and feedforward networks auto-encoders for anomaly detection can be found in
[keras_anomaly_detection/library/convolutional.py](keras_anomaly_detection/library/convolutional.py) and
[keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py) and
[keras_anomaly_detection/library/feedforward.py](keras_anomaly_detection/library/feedforward.py)

The the anomaly detection is implemented using auto-encoder with convolutional, feedforward, and recurrent networks and can be applied
to:

* timeseries data to detect timeseries time windows that have anomaly pattern
    * LstmAutoEncoder in [keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py)
    * Conv1DAutoEncoder in [keras_anomaly_detection/library/convolutional.py](keras_anomaly_detection/library/convolutional.py)
    * CnnLstmAutoEncoder in [keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py)
    * BidirectionalLstmAutoEncoder in [keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py)
* structured data (i.e., tabular data) to detect anomaly in data records
    * Conv1DAutoEncoder in [keras_anomaly_detection/library/convolutional.py](keras_anomaly_detection/library/convolutional.py)
    * FeedforwardAutoEncoder in [keras_anomaly_detection/library/feedforward.py](keras_anomaly_detection/library/feedforward.py)

# Usage

### Detect Anomaly within the ECG Data

The sample codes can be found in the [demo/ecg_demo](demo/ecg_demo).

The following sample codes show how to fit and detect anomaly using Conv1DAutoEncoder:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.convolutional import Conv1DAutoEncoder


def main():
    data_dir_path = './data'
    model_dir_path = './models'

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
    reconstruction_error = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
        reconstruction_error.append(dist)

    visualize_reconstruction_error(reconstruction_error, ae.threshold)


if __name__ == '__main__':
    main()
```

The following sample codes show how to fit and detect anomaly using LstmAutoEncoder:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.recurrent import LstmAutoEncoder


def main():
    data_dir_path = './data'
    model_dir_path = './models'
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
    reconstruction_error = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
        reconstruction_error.append(dist)

    visualize_reconstruction_error(reconstruction_error, ae.threshold)


if __name__ == '__main__':
    main()
```

The following sample codes show how to fit and detect anomaly using CnnLstmAutoEncoder:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.recurrent import CnnLstmAutoEncoder


def main():
    data_dir_path = './data'
    model_dir_path = './models'
    ecg_data = pd.read_csv(data_dir_path + '/ecg_discord_test.csv', header=None)
    print(ecg_data.head())
    ecg_np_data = ecg_data.as_matrix()
    scaler = MinMaxScaler()
    ecg_np_data = scaler.fit_transform(ecg_np_data)
    print(ecg_np_data.shape)

    ae = CnnLstmAutoEncoder()

    # fit the data and save model into model_dir_path
    ae.fit(ecg_np_data[:23, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(ecg_np_data[:23, :])
    reconstruction_error = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
        reconstruction_error.append(dist)

    visualize_reconstruction_error(reconstruction_error, ae.threshold)


if __name__ == '__main__':
    main()
```

The following sample codes show how to fit and detect anomaly using BidirectionalLstmAutoEncoder:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.recurrent import BidirectionalLstmAutoEncoder


def main():
    data_dir_path = './data'
    model_dir_path = './models'
    ecg_data = pd.read_csv(data_dir_path + '/ecg_discord_test.csv', header=None)
    print(ecg_data.head())
    ecg_np_data = ecg_data.as_matrix()
    scaler = MinMaxScaler()
    ecg_np_data = scaler.fit_transform(ecg_np_data)
    print(ecg_np_data.shape)

    ae = BidirectionalLstmAutoEncoder()

    # fit the data and save model into model_dir_path
    ae.fit(ecg_np_data[:23, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(ecg_np_data[:23, :])
    reconstruction_error = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
        reconstruction_error.append(dist)

    visualize_reconstruction_error(reconstruction_error, ae.threshold)


if __name__ == '__main__':
    main()
```

The following sample codes show how to fit and detect anomaly using FeedForwardAutoEncoder:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.feedforward import FeedForwardAutoEncoder


def main():
    data_dir_path = './data'
    model_dir_path = './models'

    # ecg data in which each row is a temporal sequence data of continuous values
    ecg_data = pd.read_csv(data_dir_path + '/ecg_discord_test.csv', header=None)
    print(ecg_data.head())
    ecg_np_data = ecg_data.as_matrix()
    scaler = MinMaxScaler()
    ecg_np_data = scaler.fit_transform(ecg_np_data)

    print(ecg_np_data.shape)

    ae = FeedForwardAutoEncoder()

    # fit the data and save model into model_dir_path
    ae.fit(ecg_np_data[:23, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(ecg_np_data[:23, :])
    reconstruction_error = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
        reconstruction_error.append(dist)

    visualize_reconstruction_error(reconstruction_error, ae.threshold)


if __name__ == '__main__':
    main()
```

# Detect Fraud in Credit Card Transaction

The sample codes can be found in the [demo/credit_card_demo](demo/credit_card_demo).

The credit card sample data is from [this repo](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras/blob/master/fraud_detection.ipynb)

Below is the sample code using FeedforwardAutoEncoder:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras_anomaly_detection.library.feedforward import FeedForwardAutoEncoder
from keras_anomaly_detection.demo.credit_card_demo.unzip_utils import unzip
from keras_anomaly_detection.library.plot_utils import plot_confusion_matrix, plot_training_history, visualize_anomaly
from keras_anomaly_detection.library.evaluation_utils import report_evaluation_metrics
import numpy as np

DO_TRAINING = False


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

    # load back the model saved in model_dir_path
    ae.load_model(model_dir_path)
    # detect anomaly for the test data
    Ypred = []
    _, Xtest, _, Ytest = train_test_split(X, Y, test_size=0.2, random_state=seed)
    reconstruction_error = []
    adjusted_threshold = 14
    anomaly_information = ae.anomaly(Xtest, adjusted_threshold)
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        predicted_label = 1 if is_anomaly else 0
        Ypred.append(predicted_label)
        reconstruction_error.append(dist)

    report_evaluation_metrics(Ytest, Ypred)
    plot_training_history(history)
    visualize_anomaly(Ytest, reconstruction_error, adjusted_threshold)
    plot_confusion_matrix(Ytest, Ypred)


if __name__ == '__main__':
    main()
```

The sample code below uses Conv1DAutoEncoder:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras_anomaly_detection.library.convolutional import Conv1DAutoEncoder
from keras_anomaly_detection.demo.credit_card_demo.unzip_utils import unzip
from keras_anomaly_detection.library.plot_utils import plot_confusion_matrix, plot_training_history, visualize_anomaly
from keras_anomaly_detection.library.evaluation_utils import report_evaluation_metrics
import numpy as np
import os

DO_TRAINING = False


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

    unzip(data_dir_path + '/creditcardfraud.zip', data_dir_path)
    csv_data = pd.read_csv(data_dir_path + '/creditcard.csv')
    estimated_negative_sample_ratio = 1 - csv_data['Class'].sum() / csv_data['Class'].count()
    print(estimated_negative_sample_ratio)
    X, Y = preprocess_data(csv_data)
    print(X.shape)

    ae = Conv1DAutoEncoder()

    training_history_file_path = model_dir_path + '/' + Conv1DAutoEncoder.model_name + '-history.npy'
    # fit the data and save model into model_dir_path
    epochs = 10
    history = None
    if DO_TRAINING:
        history = ae.fit(X, model_dir_path=model_dir_path,
                         estimated_negative_sample_ratio=estimated_negative_sample_ratio,
                         epochs=epochs)
        np.save(training_history_file_path, history)
    elif os.path.exists(training_history_file_path):
        history = np.load(training_history_file_path).item()

    # load back the model saved in model_dir_path
    ae.load_model(model_dir_path)
    # detect anomaly for the test data
    Ypred = []
    _, Xtest, _, Ytest = train_test_split(X, Y, test_size=0.2, random_state=seed)
    reconstruction_error = []
    adjusted_threshold = 10
    anomaly_information = ae.anomaly(Xtest, adjusted_threshold)
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        predicted_label = 1 if is_anomaly else 0
        Ypred.append(predicted_label)
        reconstruction_error.append(dist)

    report_evaluation_metrics(Ytest, Ypred)
    plot_training_history(history)
    visualize_anomaly(Ytest, reconstruction_error, adjusted_threshold)
    plot_confusion_matrix(Ytest, Ypred)


if __name__ == '__main__':
    main()

```


# Note

There is also an autoencoder from H2O for timeseries anomaly detection in 
[demo/h2o_ecg_pulse_detection.py](demo/ecg_demo/h2o_ecg_pulse_detection.py)

### Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.0.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 





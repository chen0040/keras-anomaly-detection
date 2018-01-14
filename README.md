# keras-anomaly-detection

Anomaly detection implemented in Keras

The the anomaly detection is implemented using auto-encoder with convolutional and recurrent networks and can be applied
to timeseries data to detect timeseries time windows that have anomaly pattern

The source codes of the recurrent and convolutional networks auto-encoders for anomaly detection can be found in
[keras_anomaly_detection/library/convolutional.py](keras_anomaly_detection/library/convolutional.py) and
[keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py) and contains the following
models:

* LSTM recurrent network: in [keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py)
* Convolutional 1D network: in [keras_anomaly_detection/library/convolutional.py](keras_anomaly_detection/library/convolutional.py)
* Bidirectional LSTM recurrent network auto-encoder: in [keras_anomaly_detection/library/recurrent.py](keras_anomaly_detection/library/recurrent.py)

# Usage





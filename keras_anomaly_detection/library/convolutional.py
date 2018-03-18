'''
Based on work by Xianshun Chen
https://github.com/chen0040/keras-anomaly-detection
'''

from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np


class Conv1DAutoEncoder(object):
    model_name = 'con1d-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.metric = None
        self.threshold = 5.0
        self.config = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
                         input_shape=(time_window_size, 1)))
        model.add(GlobalMaxPool1D())

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        print(model.summary())
        return model

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + Conv1DAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + Conv1DAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + Conv1DAutoEncoder.model_name + '-architecture.json'

    def load_model(self, model_dir_path):
        config_file_path = self.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = self.create_model(self.time_window_size, self.metric)
        weight_file_path = self.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    def fit(self, dataset, model_dir_path, batch_size=8, epochs=100, validation_split=0.1, metric='mean_absolute_error',
            estimated_negative_sample_ratio=0.9):

        self.time_window_size = dataset.shape[1]
        self.metric = metric

        input_timeseries_dataset = np.expand_dims(dataset, axis=2)

        weight_file_path = self.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = self.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = self.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        history = self.model.fit(x=input_timeseries_dataset, y=dataset,
                                 batch_size=batch_size, epochs=epochs,
                                 verbose=self.VERBOSE, validation_split=validation_split,
                                 callbacks=[checkpoint]).history
        self.model.save_weights(weight_file_path)

        scores = self.predict(dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = self.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

        return history

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)

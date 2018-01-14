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

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'

        self.time_window_size = timeseries_dataset.shape[1]
        self.metric = metric

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        print(input_timeseries_dataset.shape)

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        config_file_path = Conv1DAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

        weight_file_path = Conv1DAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = Conv1DAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = Conv1DAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=Conv1DAutoEncoder.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

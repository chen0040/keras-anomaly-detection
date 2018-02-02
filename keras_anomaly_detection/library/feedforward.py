from keras.models import Model, model_from_json
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.model_selection import train_test_split
import os
import numpy as np


class FeedForwardAutoEncoder(object):
    model_name = 'feedforward-encoder'

    def __init__(self):
        self.model = None
        self.input_dim = None
        self.threshold = None
        self.config = None

    def load_model(self, model_dir_path):
        config_file_path = FeedForwardAutoEncoder.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.input_dim = self.config['input_dim']
        self.threshold = self.config['threshold']

        architecture_file_path = FeedForwardAutoEncoder.get_architecture_file_path(model_dir_path)
        self.model = model_from_json(open(architecture_file_path, 'r').read())
        weight_file_path = FeedForwardAutoEncoder.get_weight_file_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def create_model(self, input_dim):
        encoding_dim = 14
        input_layer = Input(shape=(input_dim,))

        encoder = Dense(encoding_dim, activation="tanh",
                        activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(encoding_dim // 2, activation="relu")(encoder)

        decoder = Dense(encoding_dim // 2, activation='tanh')(encoder)
        decoder = Dense(input_dim, activation='relu')(decoder)

        model = Model(inputs=input_layer, outputs=decoder)
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        return model

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, FeedForwardAutoEncoder.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, FeedForwardAutoEncoder.model_name + '-weights.h5')

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, FeedForwardAutoEncoder.model_name + '-config.npy')

    def fit(self, data, model_dir_path, nb_epoch=None, batch_size=None, test_size=None, random_state=None,
            estimated_negative_sample_ratio=None):
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42
        if nb_epoch is None:
            nb_epoch = 100
        if batch_size is None:
            batch_size = 32
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        weight_file_path = FeedForwardAutoEncoder.get_weight_file_path(model_dir_path)
        architecture_file_path = FeedForwardAutoEncoder.get_architecture_file_path(model_dir_path)

        X_train, X_test = train_test_split(data, test_size=test_size, random_state=random_state)
        checkpointer = ModelCheckpoint(filepath=weight_file_path,
                                       verbose=0,
                                       save_best_only=True)

        self.input_dim = X_train.shape[1]
        self.model = self.create_model(self.input_dim)
        open(architecture_file_path, 'w').write(self.model.to_json())
        history = self.model.fit(X_train, X_train,
                                 epochs=nb_epoch,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 validation_data=(X_test, X_test),
                                 verbose=1,
                                 callbacks=[checkpointer]).history

        self.model.save_weights(weight_file_path)

        scores = self.predict(data)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['input_dim'] = self.input_dim
        self.config['threshold'] = self.threshold
        config_file_path = FeedForwardAutoEncoder.get_config_file_path(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

        return history

    def predict(self, data):
        target_data = self.model.predict(x=data)
        dist = np.linalg.norm(data - target_data, axis=-1)
        return dist

    def anomaly(self, data, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(data)
        return zip(dist >= self.threshold, dist)
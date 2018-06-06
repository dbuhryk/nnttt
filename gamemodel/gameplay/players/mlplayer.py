import numpy as np
import logging
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Conv3D, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Reshape
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
from keras import losses
import keras.backend as K
K.set_image_data_format('channels_last')
from gameplay.players.player import Player


class MLPlayer(Player):
    model = None

    def load_model(self, filename=None, suffix=None):
        if filename is None:
            if suffix is None:
                filename = self.__class__.__name__ + ".h5"
            else:
                filename = self.__class__.__name__ + "_" + suffix + ".h5"
        try:
            self.model.load_weights(filename)
            logging.debug("Loaded model file \"%s\"" % filename)
        except (IOError, ValueError):
            logging.warning("model file \"%s\" is missing or model structure is corrupted" % filename)

    def save_model(self, filename=None, suffix=None):
        if filename is None:
            if suffix is None:
                filename = self.__class__.__name__ + ".h5"
            else:
                filename = self.__class__.__name__ + "_" + suffix + ".h5"
        try:
            self.model.save_weights(filename)
            logging.debug("Saved model file \"%s\"" % filename)
        except IOError:
            logging.warning("Model file \"%s\" was not accessible" % filename)

    def make_game_moves(self, positions, asplayer=None):
        """
        returns valid moves for every position as player 1
        asplayer is not used
        :param positions: [?, 5, 5, 3]
        :param asplayer: makes move as player with index, normally either 1 or 2
        :return: [?, 2], where x, y = [:, 1], [:, 2]
        """
        positions = np.array(positions)  # [?, 5, 5, 3]
        predictions = MLPlayer01.predict_moves(self.model, positions)  # [?, 5, 5, 1]
        moves, weights = Player.get_best_valid_moves(predictions, positions)  # [?, n, 2], [?, n]

        #options = []
        #m = moves[weights > 0.8]
        #s = np.std(weights, axis=1, keepdims=True)

        moves = moves[:, 0, :]  # [?, 2]
        return moves

        shape = positions.shape[:-1] + (1,)
        predictions = np.random.rand(*shape)  # [?, 5, 5, 1]
        moves = Player.get_best_valid_move(predictions, positions)

    @staticmethod
    def predict_moves(model, positions):
        """
        :param model: prediction model
        :param positions: shape [?, 5, 5, 3]
        :return: [?, 5, 5, 1]
        """
        predictions = model.predict(positions, batch_size=None, verbose=0, steps=None)
        return predictions


class MLPlayer01(MLPlayer):

    def __init__(self):
        self.model = MLPlayer01.build_model([5, 5, 3])

    @staticmethod
    def build_model(input_shape):
        """
        input shape [?, N, N, 3]
        :param input_shape:
        :return:
        """
        X_input = Input(input_shape)
        X = X_input
        # Phase 1
        # X = ZeroPadding2D((1, 1))(X)
        X = Reshape((5, 5, 3, 1))(X)
        X = Conv3D(128, (2, 2, 3), strides=(1, 1, 1), name='conv0')(X)
        X = Reshape((4, 4, 128))(X)
        #X = BatchNormalization(axis=-1, name='bn0')(X)
        X = Activation('relu')(X)
        #X = MaxPooling2D((2, 2), name='max_pool0')(X)

        X = Flatten()(X)
        #X = BatchNormalization(axis=1, name='bn01')(X)
        #X = Dense(2048, activation='relu')(X)
        X = Dense(1600, activation='relu')(X)
        X = Dense(800, activation='relu')(X)
        X = Dense(400, activation='relu')(X)
        X = Dense(200, activation='relu')(X)
        X = Dense(100, activation='relu')(X)
        X = Dense(50, activation='relu')(X)
        X = Dense(25, activation='sigmoid')(X)

        #X = Reshape((5, 5, 1))(X)
        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = Model(inputs=X_input, outputs=X, name='Model')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        #model.compile(optimizer='adam', loss='mse')

        return model


class MLPlayer02(MLPlayer):

    def __init__(self):
        self.model = MLPlayer02.build_model([5, 5, 3])

    @staticmethod
    def build_model(input_shape):
        """
        input shape [?, N, N, 3]
        :param input_shape:
        :return:
        """
        X_input = Input(input_shape)
        X = X_input
        # Phase 1
        # X = ZeroPadding2D((1, 1))(X)
        # X = Reshape((5, 5, 3, 1))(X)
        # X = Conv3D(128, (2, 2, 3), strides=(1, 1, 1), name='conv0')(X)
        # X = Reshape((4, 4, 128))(X)
        # X = BatchNormalization(axis=-1, name='bn0')(X)
        # X = Activation('relu')(X)
        #X = MaxPooling2D((2, 2), name='max_pool0')(X)

        # Phase 1
        # X = ZeroPadding2D((1, 1))(X)
        #X = Conv2D(512, (3, 3), strides=(1, 1), name='conv1')(X)
        #X = BatchNormalization(axis=-1, name='bn1')(X)
        #X = Activation('relu')(X)

        X = Flatten()(X)
        #X = BatchNormalization(axis=1, name='bn01')(X)
        #X = Dense(2048, activation='relu')(X)
        #X = Lambda(lambda x: K.dropout(x, level=0.1))(X)
        #X = Dense(1600, activation='relu')(X)
        #X = Dense(800, activation='relu')(X)
        X = Dense(400, activation='relu')(X)
        X = Dense(200, activation='relu')(X)

        X = Dense(100, activation='relu')(X)
        #X = Lambda(lambda x: K.dropout(x, level=0.1))(X)
        #X = Dropout(0.1)(X)
        #X = Dense(50, activation='relu')(X)
        X = Dense(25, activation='sigmoid')(X)

        #X = Reshape((5, 5, 1))(X)
        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = Model(inputs=X_input, outputs=X, name='Model')
        #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        model.compile(optimizer='sgd', loss='categorical_crossentropy')

        #model.compile(optimizer='adam', loss='mse')

        return model


class MLPlayerSample(MLPlayer):

    def __init__(self):
        self.model = MLPlayerSample.build_model([5, 5, 3])

    @staticmethod
    def build_model(input_shape):
        """
        input shape [?, N, N, 3]
        :param input_shape:
        :return:
        """
        X_input = Input(input_shape)
        X = X_input

        #X = Flatten()(X)

        # Phase 1
        # X = ZeroPadding2D((1, 1))(X)
        # X = Conv3D(128, (2, 2), strides=(1, 1), name='conv0')(X)
        # X = BatchNormalization(axis=-1, name='bn0')(X)
        # X = Activation('relu')(X)
        #X = MaxPooling2D((2, 2), name='max_pool0')(X)

        # Phase 2
        # X = ZeroPadding2D((2, 2))(X)
        # X = Conv2D(128, (3, 3), strides=(1, 1), name='conv1')(X)
        # X = BatchNormalization(axis=-1, name='bn1')(X)
        # X = Activation('relu')(X)
        # X = MaxPooling2D((2, 2), name='max_pool1')(X)

        #X = Flatten()(X)
        #X = BatchNormalization(axis = -1)(X)
        #X = Dense(32*32*32, activation='sigmoid')(X)

        # Phase 3
        # X = ZeroPadding2D((2, 2))(X)
        # X = Conv2D(512, (3, 3), strides=(1, 1), name='conv2')(X)
        # X = BatchNormalization(axis=-1, name='bn2')(X)
        # X = Activation('relu')(X)
        # X = MaxPooling2D((2, 2), name='max_pool2')(X)

        # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
        X = Flatten()(X)
        #X = BatchNormalization(axis=1, name='bn01')(X)
        #X = Dense(2048, activation='relu')(X)
        X = Dense(1600, activation='relu')(X)
        X = Dense(800, activation='relu')(X)
        X = Dense(400, activation='relu')(X)
        X = Dense(200, activation='relu')(X)
        X = Dense(100, activation='relu')(X)
        X = Dense(50, activation='relu')(X)
        #X = Lambda(lambda x: K.dropout(x, level=0.2))(X)
        #X = Dropout(0.1)(X)
        #X = Dense(128, activation='relu')(X)
        X = Dense(25, activation='sigmoid')(X)

        #X = Reshape((5, 5, 1))(X)
        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = Model(inputs=X_input, outputs=X, name='Model')
        #model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        #model.compile(optimizer='adam', loss='mse')

        return model
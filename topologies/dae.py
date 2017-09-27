""" Denoising Auto-Encoder """

import keras.backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Activation, Reshape, Flatten


MODEL_CONV_FILTERS = 8
MODEL_CONV_KERNEL_SIZE = 4
MODEL_CONV_STRIDES = 1
MODEL_CONV_PADDING = 'same'


def build_model(input_shape):
    seq_length = input_shape[0]

    # build it!
    model = Sequential()

    # conv
    model.add(Conv1D(input_shape=input_shape,
                     filters=MODEL_CONV_FILTERS,
                     kernel_size=MODEL_CONV_KERNEL_SIZE,
                     strides=MODEL_CONV_STRIDES,
                     padding=MODEL_CONV_PADDING))
    model.add(Activation('linear'))

    # reshape
    model.add(Flatten())

    # dense
    model.add(Dense(units=seq_length*MODEL_CONV_FILTERS))
    model.add(Activation('relu'))

    # dense
    model.add(Dense(units=128))
    model.add(Activation('relu'))

    # dense
    model.add(Dense(units=seq_length*MODEL_CONV_FILTERS))
    model.add(Activation('relu'))

    # reshape
    model.add(Reshape(target_shape=(seq_length, MODEL_CONV_FILTERS)))

    # conv
    model.add(Conv1D(filters=1,
                     kernel_size=MODEL_CONV_KERNEL_SIZE,
                     strides=MODEL_CONV_STRIDES,
                     padding=MODEL_CONV_PADDING))
    model.add(Activation('linear'))

    # define accuracy
    ON_POWER_THRESHOLD = 10
    def acc(y_true, y_pred):
        return K.mean(K.equal(K.greater_equal(y_true, ON_POWER_THRESHOLD),
                              K.greater_equal(y_pred, ON_POWER_THRESHOLD)))

    # compile it!
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae', acc])

    return model

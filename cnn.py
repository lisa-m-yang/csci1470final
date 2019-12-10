import sys
import numpy as np
import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, in_channels, out_channels):

        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 5

        self.model = Sequential()
        self.model.add(Conv1D(self.out_channels, self.kernel_size, activation='relu', input_shape=(None, self.in_channels))


    def call(self, inputs):
        output = self.model(inputs)

        return tf.reduce_max(output, 2)[0]

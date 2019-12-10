import sys
import numpy as np
import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, filters):

        super(CNN, self).__init__()

        self.filters = filters
        self.kernel_size = 5

        self.model = Sequential()
        self.model.add(Conv1D(self.out_channels, self.kernel_size, activation='relu'))


    def call(self, inputs):
        output = self.model(inputs)

        return output

import tensorflow as tf
from tensorflow.keras.layers import Dense

class Highway(tf.keras.Model):
    def __init__(self, size):

        super(Highway, self).__init__()

        self.size = size

        self.gate = Dense(self.size, activation='sigmoid', input_shape=(self.size,))
        self.proj = Dense(self.size, activation='relu')

    def call(self, inputs):
        gate_output = self.gate(inputs)
        proj_output = self.proj(inputs)

        return gate_output * proj_output + (1 - gate_output) * proj_output

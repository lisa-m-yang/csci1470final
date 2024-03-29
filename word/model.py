import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


class RNN_WORD_Model(tf.keras.Model):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, vocab_size):
        super(RNN_WORD_Model, self).__init__()

        # initialize hyperparamters

        self.vocab_size = vocab_size
        self.dropout_size = 0.2
        self.window_size = 20
        self.embedding_size = 650
        self.hidden_size = 650
        self.batch_size = 20

        self.dropout = tf.keras.layers.Dropout(self.dropout_size)
        self.encoder = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.RNN1 = tf.keras.layers.LSTM(self.hidden_size, dropout=self.dropout_size, return_sequences=True, return_state=True)
        self.RNN2 = tf.keras.layers.LSTM(self.hidden_size, dropout=self.dropout_size)
        self.decoder = tf.keras.layers.Dense(self.vocab_size, activation='softmax', kernel_initializer='random_uniform')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        self.loss_function = tf.keras.losses.BinaryCrossentropy()


    def call(self, input, hidden):

        emb = self.dropout(self.encoder(input))
        output, hidden1, hidden2 = self.RNN1(emb, initial_state=hidden)
        output, hidden1, hidden2 = self.RNN2(emb, initial_state=(hidden1, hidden2))
        output = self.dropout(output)
        decoded = self.decoder(output)

        return decoded, (hidden1, hidden2)

    def init_hidden(self):
        """
        Return a Variable filled with zeros based on the data shape.
        """

        return (tf.zeros([2, self.batch_size, self.hidden_size]), tf.zeros([2, self.batch_size, self.hidden_size]))

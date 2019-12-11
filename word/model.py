import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose


class RNN_WORD_Model(tf.keras.Model):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, vocab_size):
        super(RNN_WORD_Model, self).__init__()
        
        # initialize hyperparamters

        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 650
        self.batch_size = 64
        self.rnn_size = 256
	self.nlayer = 2
	self.nhid = 650
        
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.encoder = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
      
        self.LSTM = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)        
        self.decoder = tf.keras.layers.Dense(self.vocab_size, activation='softmax', kernel_initializer='random uniform') #[hiddensz, vocabsz]        
        # how to set min and max -.1/.1 within initializer?
	self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


        

    def call(self, input, hidden):
        emb = self.dropout(self.encoder(input))
        output, hidden1, hidden2 = self.LSTM(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, (hidden1, hidden2)


    def init_hidden(self, bsz):
		"""
		Return a Variable filled with zeros based on the data shape.
		"""
	hidden = (tf.zeros_like(self.nlayers, self.batch_size, self.nhid),
		tf.zeros_like(self.nlayers, self.batch_size, self.nhid))
	return hidden
		



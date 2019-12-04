import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose


class Model(tf.keras.Model):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, vocab_size):
        super(Model, self).__init__()
        
        # initialize vocab_size, embedding_size

        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 650
        self.batch_size = 64
        self.rnn_size = 256
        
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.encoder = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
      
        self.LSTM = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)        
        self.decoder = tf.keras.layers.Dense(self.vocab_size, activation='softmax') #[hiddensz, vocabsz]        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


        

    def forward(self, input, hidden):
        emb = self.dropout(self.encoder(input))
        output, hidden = self.LSTM(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    # unsure
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
        
class Position_Encoding_Layer(tf.keras.layers.Layer):
	def __init__(self, window_sz, emb_sz):
		super(Position_Encoding_Layer, self).__init__()
		self.positional_embeddings = self.add_weight("pos_embed",shape=[window_sz, emb_sz])

	@tf.function
	def call(self, x):
		"""
		Adds positional embeddings to word embeddings.

		:param x: [BATCH_SIZE x WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
		:return: [BATCH_SIZE x WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
		"""
		return x+self.positional_embeddings

class Transformer(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################
        
        # Define hyperparameters
        self.batch_size = 100
		self.embedding_size = 32
		self.rnn_size = 256
        # Define embeddings, encoder, decoder, and feed forward layers
        self.E1 = tf.keras.layers.Embedding(self.french_vocab_size, self.embedding_size, input_length=self.french_window_size)
		self.E2 = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size, input_length=self.english_window_size)
        self.enc_positional_encoding = transformer.Position_Encoding_Layer(self.french_window_size, self.embedding_size)
		self.dec_positional_encoding = transformer.Position_Encoding_Layer(self.english_window_size, self.embedding_size)
        
        # TODO: Define encoder and decoder layers:
		self.encoder = transformer.Transformer_Block(self.embedding_size,is_decoder=False,multi_headed=False)
		self.decoder = transformer.Transformer_Block(self.embedding_size,is_decoder=True,multi_headed=False)

		# Define dense layer(s)
		self.Dense1 = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')

		# Define optimizer
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
    @tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""

		# TODO:
		#1) Add the positional embeddings to french sentence embeddings
		encoder_input = self.E1(encoder_input)
		encoder_input = self.enc_positional_encoding(encoder_input)

		#2) Pass the french sentence embeddings to the encoder
		encoder_output = self.encoder(encoder_input)

		#3) Add positional embeddings to the english sentence embeddings
		decoder_input = self.E2(decoder_input)
		decoder_input = self.dec_positional_encoding(decoder_input)

		#4) Pass the english embeddings and output of your encoder, to the decoder
		decoder_output = self.decoder(decoder_input, encoder_output)

		#5) Apply dense layer(s) to the decoder out to generate probabilities
		prbs = self.Dense1(decoder_output)
		# prbs = self.Dense2(prbs)

		return prbs


        

# ========= torch implementation ========== #
class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

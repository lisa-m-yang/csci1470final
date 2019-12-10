import sys
import numpy as np
import tensorflow as tf

class NLG(tf.keras.Model):
	def __init__(self, speakers, embed_size, hidden_size, dropout_rate, vocab, no_char_decoder, learning_rate, clip_grad, lr_decay):
		"""
    Natural Language Generation model using a Neural Machine Translation context 
		"""
		super(NLG, self).__init__()

		self.NMT_speakers = []
    self.NMT_models = []
    self.NMT_optimizers = []
    self.clip_grad = clip_grad
    self.lrs = []
    self.lr_decay = lr_decay
    
    for speaker in speakers:
      model = NMT(embed_size=embed_size, hidden_size=hidden_size, dropout_rate=dropout_rate, vocab=vocab, no_char_decoder=no_char_decoder)
      optimizer = tf.keras.optimizers.Adam(learning_rate)
      self.NMT_speakers.append(speaker.replace("/", "-").replace(" ","-"))
      self.NMT_models.append(model)
      self.NMT_optimizers.append(optimizer)
      self.lrs.append(lr)

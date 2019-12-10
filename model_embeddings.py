import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout

from cnn import CNN
from highway import Highway

class ModelEmbeddings(tf.keras.Model):
    def __init__(self, embedding_size, vocab):

        super(ModelEmbeddings, self).__init__()

        self.char = 50
        self.embedding_size = embedding_size
        self.vocab = vocab

        self.embeddings = Embedding(self.vocab_size, self.char)
        self.cnn = CNN(self.char, self.embedding_size)
        self.highway = Highway(self.embedding_size)
        self.dropout = Dropout(0.3)


    def call(self, inputs):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary
        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        sentence_length = inputs.size()[0]
        batch_size = inputs.size()[1]
        e = self.embeddings(inputs)

        e = e.permute(0, 1, 3, 2)
        e = e.contiguous()
        e = e.view(-1, e.size()[2], e.size()[3])

        x_conv_out = self.cnn(e)

        x_highway = self.highway(x_conv_out)

        x_word_emb = self.dropout(x_highway)

        x_word_emb = x_word_emb.view(sentence_length, batch_size, self.embed_size)
        
        return x_word_emb

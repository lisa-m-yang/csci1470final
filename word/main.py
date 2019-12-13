import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from preprocess import get_num_lines, get_data, split_data, build_vocab, convert_to_id
from model import RNN_WORD_Model


def train(model, train_data):
    """
    Trains the model for one epoch.
    """
    length = len(train_data)
	num_batches = int(length/model.batch_size)
    hidden = model.init_hidden()
	for i in range(num_batches):
		data = train_data[i*model.batch_size:(i+1)*model.batch_size]
        targets = train_data[i*model.batch_size+1:(i+1)*model.batch_size+1]
		with tf.GradientTape() as tape:
			output, hidden = model.call(data, hidden)
			loss = model.loss_function(output, targets)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer = model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_data):
	"""
	Runs through one epoch - all testing examples.

	:returns: perplexity of the test set
	"""
    total_loss = 0
    length = len(test_data)
	num_batches = int(length/model.batch_size)
    hidden = model.init_hidden()
	for i in range(num_batches):
		data = test_data[i*model.batch_size:(i+1)*model.batch_size]
        targets = test_data[i*model.batch_size+1:(i+1)*model.batch_size+1]
		output, hidden = model.call(data, hidden)
		total_loss = model.loss_function(output, targets)
    perplexity = np.exp(total_loss/num_batches)

    return perplexity


def main():
    NAME_IDX = 5
    data_file = '../data/the_office_scripts.csv'

    length = get_num_lines(data_file)
    data_names = get_data(data_file, NAME_IDX, length)
    vocab = build_vocab(data_names)
    data_ids = convert_to_id(vocab, data_names)
    train_data, test_data = split_data(data_ids, length)

    num_tokens = len(vocab)
    model = RNN_WORD_Model(num_tokens)

    for i in range(40):
        train(model, train_data)

    test(model, test_data)

if __name__ == '__main__':
    main()

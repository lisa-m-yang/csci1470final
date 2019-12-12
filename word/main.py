import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import Model
import preprocess
from model import RNN_WORD_Model


def train(model, train_data):
    """
    Trains the model for one epoch.
    """
    total_loss = 0
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        with tf.GradientTape() as tape:
            output, hidden = model(data, hidden)
            loss = loss_function(prbs=output, labels=targets)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss

def test(model, test_data):
	"""
	Runs through one epoch - all testing examples.


	:returns: perplexity of the test set, per symbol accuracy on test set
	"""
	curr_loss = 0
	step = 0
	perplexity = 0
	per_symbol_accuracy = 0
	hidden = model.init_hidden(args.batch_size)

	for start, end in zip(range(0, len(test_data), args.batch_size),
						  range(args.batch_size, len(test_data),
								args.batch_size)):


		with tf.GradientTape() as tape:
			prbs = model(encoder_input=test_french2, decoder_input=test_english2)
			loss = loss_function(prbs=prbs, labels=test_labels2, mask=mask)
			# print("Loss", loss/model.batch_size, step)
			curr_loss += loss
		perplexity = tf.math.exp((curr_loss / (step * sum_mask)))
		per_symbol_accuracy = sum_mask * model.accuracy_function(prbs, test_labels2, mask)
		step += 1

	return perplexity, per_symbol_accuracy


def main():
    NAME_IDX = 5
    data_file = '../data/the_office/the_office_scripts.csv'

    length = get_num_lines(data_file)
    data_names = get_data(data_file, NAME_IDX, length)
    vocab = build_vocab(data)
    data_ids = convert_to_id(vocab, data_names)
    train_data, valid_data, test_data = split_data(data_ids)

    num_tokens = len(vocab)
    model = RNN_WORD_Model(num_tokens)

    for i in range(40):
        train(model, train_data)
        test(model, valid_data)

    test(model, test_data)

if __name__ == '__main__':
    main()

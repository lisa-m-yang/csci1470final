import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import Model
import preprocess


parser = argparse.ArgumentParser(description='RNN Language Model')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net ')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')



args = parser.parse_args()

###############################################################################
# Load data and batch.
###############################################################################

corpus = preprocess.Corpus(args.data)

###############################################################################
# Build the model: define token length, initialize model, define loss.
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

loss_function = tf.keras.losses.BinaryCrossentropy()

def train(model, train_data):
    """
    Trains the model for one epoch.
    """
    curr_loss = 0
    step = 0
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        with tf.GradientTape() as tape:
            output, hidden = model(data, hidden)
            loss = loss_function(prbs=output, labels=targets)
        gradients = tape.gradient(loss, model.trainable_variables)
	model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
	step += 1
	
def test(model, test_data):
	"""
	Runs through one epoch - all testing examples.

	
	:returns: perplexity of the test set, per symbol accuracy on test set
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!

	curr_loss=0
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

def export(path, batch_size, seq_len):
	input_holder = tf.zeros(seq_len * batch_size)
	hidden = model.init_hidden(batch_size)
	# TODO: write model, input_holder, hidden to a path for exporting

def main():
    NAME_IDX = 5
    data_file = '../data/the_office/the_office_scripts.csv'

    num_lines = get_num_lines(data_file)
    data = get_data(data_file, num_lines)
    train_names = data[:int(0.8 * length)]
    valid_names = data[int(0.8 * length):int(0.9 * length)]
    test_names = data[int(0.9 * length):]
    vocab = build_vocab(data)
    train_ids = convert_to_id(vocab, train_names)
    valid_ids = convert_to_id(vocab, valid_names)
    test_ids = convert_to_id(vocab, test_names)

if __name__ == '__main__':
    main()

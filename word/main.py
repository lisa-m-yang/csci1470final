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

corpus = word_preprocess.Corpus(args.data)

def batchify (data, bsz):
    # find how many batches we have
    nbatch = data.size(0) // bsz
    # find remainder, trim ends
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model: define token length, initialize model, define loss.
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = tf.keras.losses.BinaryCrossentropy()

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors to detach them from history.
    """
    if isinstance(h, tf.tensor):
        # unsure
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

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
            loss = criterion(prbs=output, labels=targets)
        gradients = tape.gradient(loss, model.trainable_variables)
	model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
	step += 1

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

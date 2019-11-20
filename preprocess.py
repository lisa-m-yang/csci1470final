import tensorflow as tf
import numpy as np


# reads .csv files and converts them into text files


def get_data(train_file):
    """
    Read and parse the train file line by line, then tokenize the sentences 
    to build the train data.
    Create a vocabulary dictionary that maps all the unique tokens from your train 
    data as keys to a unique integer value.
    Then vectorize your train data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """

    # load and concatenate training data from training file.
    # read in and tokenize training data

    with open(train_file, 'r') as f:
        words = f.read().split()

    vocab = set(" ".join(words).split())
    word2id = {w: i for i, w in enumerate(list(vocab))}  # label encode

    training_data = [word2id[x] for x in words]

    # return tuple of training tokens, testing tokens, and the vocab dictionary.


    return training_data, word2id

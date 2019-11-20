import tensorflow as tf
import numpy as np


# reads .csv files and converts them into text files
def csv2txt (csv_file):
    csv_file = 
    txt_file = 
    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()


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

    with open(train_file) as train_text:
        train_data = train_text.read().split()
        vocabulary = {w: i for i, w in enumerate(list(set(train_data)))}
    
    train = [vocabulary[w] for w in train_data]
    # return tuple of training tokens, testing tokens, and the vocab dictionary.

    return train, vocabulary

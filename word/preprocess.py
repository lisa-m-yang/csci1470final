import csv
import numpy as np
from io import open

def get_num_lines(input_file):
    """
    Calculate the number of lines in CSV file

	:param input_file:  file path for CSV
	:return: num_lines:  the number of lines in the CSV
    """
    with open(input_file, encoding='ISO-8859-1') as input:
        reader = csv.reader(input)
        next(reader)
        num_lines = sum(1 for row in reader)

    return num_lines

def get_data(input_file, idx, num_lines):
    """
    Get the data of a column in CSV file

	:param input_file:  file path for CSV
    :param idx:  index for the column in CSV
    :param num_lines:  the number of lines in the CSV
	:return: data:  an array containing the column data
    """
    with open(input_file, encoding='ISO-8859-1') as input:
        reader = csv.reader(input)
        next(reader)

        data = [row[idx].strip() for i, row in enumerate(reader)]

    return data

def split_data(data, length):
    """
    Split all data into train, valid, test data

	:param data:  array containing all the data
    :param length:  length of data
	:return: train, test:  a tuple of data for train and test
    """
    train = data[:int(0.8 * length)]
    test = data[int(0.8 * length):]

    return train, test

def build_vocab(names):
	"""
    Builds vocab from list of words

	:param names:  list of words, each a name
	:return: dictionary: word --> unique index
    """
	all_names = sorted(list(set(names)))

	vocab =  {word:i for i,word in enumerate(all_names)}

	return vocab

def convert_to_id(vocab, names):
	"""
    Convert names to indexed

	:param vocab:  dictionary, word --> unique index
	:param names:  lists of words, each representing a name
	:return: numpy array of integers, with each representing the word indeces in the corresponding names
    """
	return np.stack([vocab[name] for name in names if name in vocab])

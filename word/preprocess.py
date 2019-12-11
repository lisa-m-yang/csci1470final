import numpy as np
from io import open

def get_num_lines(input_file):
    with open(input_file, encoding='ISO-8859-1') as input:
        reader = csv.reader(input)
        next(reader)
        num_lines = sum(1 for row in reader)

    return num_lines

def get_data(input_file, num_lines):
    with open(input_file, encoding='ISO-8859-1') as input:
        reader = csv.reader(input)
        next(reader)

        data = [row[NAME_IDX].strip() for i, row in enumerate(reader)]

    return data

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
	return np.stack([vocab[word] if word in vocab for name in names])

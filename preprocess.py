import tensorflow as tf
import numpy as np
import math

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
START_TOKEN = "*START*"
STOP_TOKEN = "*STOP*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 14
##########DO NOT CHANGE#####################

def pad_sents_char(sentences, char_pad_token):
    """
    Pad list of sentences according to the longest sentence in the batch and max_word_length.
    
    :param sentences (list[list[list[int]]]): list of sentences
    :param char_pad_token (int): index of the character-padding token
    :return padded_sentences (list[list[list[int]]]): 
        list of padded sentences such that each sentence in the batch now 
        has same number of words and each word has an equal number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # words longer than 21 characters should be truncated
    max_word_length = 21
    max_sent_length = max(len(s) for s in sentences)
    
    padded_sentences = []
    for s in sentences:
        padded_words = []
        for w in s:
            padded_w = [char_pad_token] * max_word_length
            padded_w[:(len(w))] = w[:max_word_length]
            padded_words.append(padded_w)
        while len(padded_words) != max_sent_length:
            padded_words.append([char_pad_toekn] * max_word_length)
        padded_sentences.append(padded_words)
    
    return padded_sentences

def pad_sents(sentences, pad_token):
    """
    Pad list of sentences according to the longest sentence in the batch.
    
    :param sentences (list[list[int]]): list of sentences
    :param pad_token (int): padding token
    :return padded_sentences (list[list[int]]): 
        list of padded sentences such that each sentence in the batch now 
        has same number of words
        Output shape: (batch_size, max_sentence_length)
    """
    # words longer than 21 characters should be truncated
    max_length = max(len(s) for s in sentences)
    
    padded_sentences = []
    for s in sentences:
        padded = [pad_token] * max_length
        padded[:len(s)] = s
        padded_sentences.append(padded)
    
    return padded_sentences
    
def read_data(file_path, language):
    """
    Reads a text file, where sentence is separated by a newline character. 
    
    :param file_path (str): path to file containing the corpus
    :param langauge (str): 
        "source" or "target" indicating whether the corpus 
        is of the source or target language
    :return data (list[list[str]]): list of sentences
    """
    data = []
    with open(file_path) as data_file:
		for line in data_file: 
            sentence = line.strip().split(' ')
            # append start and stop tokens to target sentence
            if language == 'target':
                sentence = [START_TOKEN] + sentence + [STOP_TOKEN]
            data.append(sentence)
            
	return data

def read_data_nlg(file_path):
    """
    Reads a text file, where sentence is separated by a newline character. 
    
    :param file_path (str): path to file containing the corpus
    :return speakers (list[str]]), source (list[list[str]]), target (list[list[str]]): 
        speakers, a list of speakers
        source, a list of sentences in the source language
        target, a list of sentences in the target language
    """
    speakers = []
    source = []
    target = []
    prevLine = None
    with open(file_path) as data_file:
		for line in data_file: 
            if prevLine != None:
                sentence = line.strip().replace("\n", "").split(' ')
                i = 0
                while ":" not in sent[i]:
                    i += 1
                    speaker = sentence[i][:-1]
                if speaker not in speakers:
                    speakers.append(speaker)
                sentence = [START_TOKEN] + sentence + [STOP_TOKEN]
                target.append(sentence)
                source.append(prevLine)
                prevLine = sentence[1:-1]
            else:
                sentence = line.strip().replace("\n", "").split(' ')
                speaker = sent[0][:-1]
                if speaker not in speakers:
                    speakers.append(speaker)
                prevLine = sent
            sentence = line.strip().split(' ')
            # append start and stop tokens to target sentence
            if language == 'target':
                sentence = [START_TOKEN] + sentence + [STOP_TOKEN]
            data.append(sentence)
            
	return speakers, source, target

# reads .csv files and converts them into text files
def csv2txt (csv_file):
    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()
        

def build_vocab(sentences):
	"""
	FROM HW 4

  Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	FROM HW 4

  Convert sentences to indexed

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

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

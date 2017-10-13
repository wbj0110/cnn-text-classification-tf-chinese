from collections import Counter

import numpy as np
import re
import codecs
import itertools
import os

def load_data(train_data_dir):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(train_data_dir)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def load_data_test(vocabulary, sequence_length):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    x_raw, sentences, labels = load_data_and_labels_test()
    sentences_padded = pad_sentences(sentences, sequence_length=sequence_length)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)


    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x_raw, x, y, vocabulary]


def load_data_and_labels_test():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(codecs.open("./data/chinese/test/pos.txt", "r", "utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(codecs.open("./data/chinese/test/neg.txt", "r", "utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_raw = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [list(s.strip()) for s in x_raw if(s.strip() !='')]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_raw, x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>", sequence_length=None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if (sequence_length is None):
        sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if(num_padding < 0):
            new_sentence = sentence[:sequence_length]
        else:
            new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    # x = np.array([[vocabulary[word] for word in sentence if(word in vocabulary)] for sentence in sentences])
    x = np.array([[vocabulary.get(word,0) for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(train_data_dir=None):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    path_list = []
    for file in os.listdir(train_data_dir):
        path = os.path.join(train_data_dir, file)
        if os.path.isfile(path):
            path_list.append(path)

    print(path_list)


    path_list


            # Load data from files
    positive_examples = list(codecs.open("./data/chinese/pos.txt", "r", "utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(codecs.open("./data/chinese/neg.txt", "r", "utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [list(s) for s in x_text if(s.strip() !='')]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

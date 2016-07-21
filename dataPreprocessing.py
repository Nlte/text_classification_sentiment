"""Data processing module."""
import re
import random
from nltk.corpus import stopwords
import logging
import numpy as np
import nltk
import sys as Sys
import gensim
import string

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def toWordList(review, rm_stopwords=False, rm_nalpha=False, lang="english"):
    """Return a list of words from a sentence."""
    txt = review
    words = []
    if rm_nalpha:
        regex = re.compile('[%s]' % re.escape(string.punctuation.replace("'", "")))
        txt = regex.sub(' ', txt)
        txt = re.sub(' +', ' ', txt)
    words = txt.lower().split()
    if rm_stopwords:
        exclude = set(stopwords.words(lang))
        words = [w for w in words if w not in exclude]
    return words


def toSentenceList(review,
                   rm_stopwords=False, rm_nalpha=False, lang="english"):
    """Transform a review into a list of sentences(=list of words)."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for r_sent in raw_sentences:
        if len(r_sent) > 0:
            sentences.append(toWordList(r_sent, rm_stopwords, rm_nalpha, lang))
    return sentences


def buildVectors(sentences, labels, w2v_model=None):
    """Build vectors with Google model."""
    # x : list of sentences / 1 sentence : list of word index
    # notinvocab : word that doesnt belong to Google vector space
    notinvocab = []
    x = []

    # Loading model
    if w2v_model:
        model = gensim.models.Word2Vec.load(w2v_model)
    else:
        model = ('model/300features-GoogleNews.w2v')

    print("Building w2v vectors ...")
    l = len(sentences)
    i = 0
    printProgress(i, l, prefix='Progress:', suffix='Complete', barLength=50)

    for sentence in sentences:
        buffer_s = []
        for word in sentence:
            try:
                # .tolist() cast numpy datatype into python std list
                buffer_s.append(model[word].tolist())
            except:
                notinvocab.append(word)
        x.append(buffer_s)
        i += 1
        printProgress(i, l, prefix='Progress:', suffix='Complete', barLength=50)
    # y : list of sentiments for each sentence
    y = labels
    # Free memory allocation for Google news model
    del model
    return x, y, notinvocab


def padSentences(sentences, pad_word, max_len=None):
    """Pad sentences to match shape of the neural net input (ConvLayer)."""
    if max_len:
        max_length = max_len
    else:
        max_length = max(len(x) for x in sentences)
    # max_length = 2488
    padded_sentences = []
    l = len(sentences)
    i = 0
    printProgress(i, l, prefix='Progress:', suffix='Complete', barLength=50)
    for i in range(len(sentences)):
        sent = sentences[i]
        nb_pad = max_length - len(sent)
        new_sent = sent + [pad_word] * nb_pad
        padded_sentences.append(new_sent)
        i += 1
        printProgress(i, l, prefix='Progress:', suffix='Complete', barLength=50)
    return padded_sentences


def printProgress(iteration, total, prefix='', suffix='', decimals=2, barLength=100):

    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    Sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    Sys.stdout.flush()
    if iteration == total:
        print("\n")


def shuffle_lists(list1, list2):
    indexes = list(range(len(list1)))
    random.shuffle(indexes)
    list1_shuf = []
    list2_shuf = []
    for index in indexes:
        list1_shuf.append(list1[index])
        list2_shuf.append(list2[index])
    return list1_shuf, list2_shuf


def buildInputData_debug():
    """Create dataset for the neural network."""
    # Loading training dataset
    print("Loading data ...")
    raw = ["Good product for the price you won't find anything better.",
           "It could have been a good movie, actors are really interesting but they failed the end.", "Awful guitar, the strings hurt my fingers."]
    raw_test = ["Iphone 6 is a really good product", "Very impressed with this guitar for the price."]
    x_train = []
    x_test = []
    for sentence in raw:
        x_train.append(toWordList(sentence, rm_nalpha=True))
    for sentence in raw_test:
        x_test.append(toWordList(sentence, rm_nalpha=True))
    print(x_train)
    print(x_test)
    labels_train = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    labels_test = [[0.0, 1.0], [0.0, 1.0]]
    model = ""
    x_train, y_train, notinvocab = buildVectors(x_train, labels_train, w2v_model=model)  # ok
    x_test, y_test, notinvocab2 = buildVectors(x_test, labels_test, w2v_model=model)
    x = x_train + x_test
    x = padSentences(x, np.zeros(300, dtype='float32').tolist())
    # x_train = padSentences(x_train, np.zeros(300, dtype='float32').tolist())  # ok
    # x_test = padSentences(x_test, np.zeros(300, dtype='float32').tolist(), max_len=12)
    x_train = x[0:3]
    print(len(x_train))
    x_test = x[3:5]
    print(len(x_test))
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def buildInputData(filename, w2v_model, n_classes=2, lang='english'):
    """Build the input data for the neural net."""
    with open(filename, 'r') as f:
        raw = f.read()
        raw.encode('utf-8')
    sentences_list = toSentenceList(raw, rm_nalpha=True, lang=lang)
    sentences_list_str = []
    for sentence in sentences_list:
        sentences_list_str.append(" ".join(sentence))
    sentences_list = sentences_list[:1000]
    sentences_list_str = sentences_list_str[:1000]
    if n_classes == 2:
        y_buffer = [[0.0, 0.0] for s in sentences_list]
    elif n_classes == 3:
        y_buffer = [[0.0, 0.0, 0.0] for s in sentences_list]
    x, y, errors = buildVectors(sentences_list, y_buffer, w2v_model=w2v_model)
    x = padSentences(x, np.zeros(300, dtype='float32').tolist())
    x = np.array(x)
    y = np.array(y)
    return x, y, sentences_list_str


def buildInputData_twitter(filename, w2v_model, n_classes=2, lang='english'):
    """Build the input data for the neural net from twitter."""
    sentences_list = []
    sentences_list_str = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sentences_list.append(toWordList(line, rm_nalpha=True, lang=lang))
            sentences_list_str.append(line)
    sentences_list = sentences_list[:1000]
    sentences_list_str = sentences_list_str[:1000]
    if n_classes == 2:
        y_buffer = [[0.0, 0.0] for s in sentences_list]
    elif n_classes == 3:
        y_buffer = [[0.0, 0.0, 0.0] for s in sentences_list]
    x, y, errors = buildVectors(sentences_list, y_buffer, w2v_model=w2v_model)
    x = padSentences(x, np.zeros(300, dtype='float32').tolist())
    x = np.array(x)
    y = np.array(y)
    return x, y, sentences_list_str


def batch_iterator(data, batch_size, num_epochs, shuffle=True):
    """Generates a batch iterator for a dataset."""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
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

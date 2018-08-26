import csv
from termcolor import colored
from gensim import corpora
from gensim import models
from pymystem3 import Mystem
import json
import sys
import os
import logging
import gc
from tqdm import tqdm
from nltk.corpus import stopwords as nltk_stopwords
from stop_words import get_stop_words as py_stopwords
from itertools import islice
import re
from number_utils import _num2words, _ordinal_to_cardinal
import pandas as pd
import utils
from multiprocessing import Pool
import math
import numba
import dask.dataframe as dd
from dask.multiprocessing import get


csv.field_size_limit(sys.maxsize)
stemmer = Mystem()  # already has caching
stopwords = None
pattern = re.compile('[^\w ]')
NUM_ROWS_TO_PROCESS = 582167
ROWS_BUFFER_SIZE = 10
CORE_NUM = 8
BLOCK_SIZE = 2e9


def get_stopwords():
    global stopwords
    if stopwords is None:
        stopwords = nltk_stopwords.words('russian')
        with open('./stopwords.txt', 'r') as f:
            stopwords.extend(map(lambda x: x.replace('\n', ''), f.readlines()))
        stopwords = set(stopwords)
    return stopwords


def normalize(text):
    global pattern, stopwords
    text = pattern.sub('', _str(text).lower())
    text = ' '.join(map(_num2words, filter(lambda x: len(x) > 0, text.split())))
    text = ''.join(stemmer.lemmatize(text))
    text = ' '.join(map(_ordinal_to_cardinal, filter(lambda x: x not in stopwords, text.split())))
    return text


def process(text):
    global pattern, stopwords
    _stopwords = stopwords
    text = pattern.sub('', text.lower())
    text = ' '.join(map(_num2words, filter(lambda x: len(x) > 0, text.split())))
    text = ''.join(stemmer.lemmatize(text))
    words = list(map(_ordinal_to_cardinal, filter(lambda x: x not in _stopwords, text.split())))
    bigrams = list(map(lambda x: '\t'.join(x), zip(words[:-1], words[1:])))
    return [words, bigrams]


def _join_fn(x):
    return ' '.join(x)


def _str(x):
    if isinstance(x, str):
        return x
    return str(x)


def _process(x):
    return process(' '.join([_str(x[1]), _str(x[2])]))


def _process_df(df):
    return df.apply(_process, axis=1)


def _join_df(df):
    return df.apply(_join_fn, axis=0)


def get_words_bigrams(text):
    words = _str(text).split()
    bigrams = list(map(lambda x: '\t'.join(x), zip(words[:-1], words[1:])))
    return words, bigrams


def _update_dicts(x, dict1, dict2):
    words, bigrams = get_words_bigrams(x)
    dict1.add_documents([words])
    dict2.add_documents([bigrams])
    return 0


def bigramize(words):
    bigrams = list(map(lambda x: '\t'.join(x), zip(words[:-1], words[1:])))
    return bigrams


def get_dictionaries():
    if not (os.path.exists('dict_1.gensim') and os.path.exists('dict_2.gensim')):
        # dictionary_1 = corpora.Dictionary()
        dictionary_2 = corpora.Dictionary()
        for doc_part in tqdm(os.listdir('./docs/')):
            tsv_path = os.path.join('./docs/', doc_part)
            with open(tsv_path, 'r') as f:
                pass
                # dictionary_1.add_documents((re.sub('[^а-яА-Яa-zA-z ]', '', line).split() for line in f))
            with open(tsv_path, 'r') as f:
                dictionary_2.add_documents((bigramize(re.sub('[^а-яА-Яa-zA-z ]', '', line).split()) for line in f))

        # dictionary_1.save('dict_1.gensim')
        dictionary_2.save('dict_2.gensim')
    # dictionary_1 = corpora.Dictionary.load('dict_1.gensim')
    dictionary_2 = corpora.Dictionary.load('dict_2.gensim')
    print(dictionary_2)
    # return dictionary_1, dictionary_2


if __name__ == '__main__':
    get_stopwords()
    get_dictionaries()

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


csv.field_size_limit(sys.maxsize)
stemmer = Mystem()  # already has caching
stopwords = None
pattern = re.compile('[^\w ]')
NUM_ROWS_TO_PROCESS = 582167
ROWS_BUFFER_SIZE = 1000
CORE_NUM = 8


def get_stopwords():
    global stopwords
    if stopwords is None:
        stopwords = nltk_stopwords.words('russian')
        with open('./stopwords.txt', 'r') as f:
            stopwords.extend(map(lambda x: x.replace('\n', ''), f.readlines()))
        stopwords = set(stopwords)
    return stopwords


def process(text):
    global pattern
    _stopwords = get_stopwords()
    text = pattern.sub('', text.lower())
    text = ' '.join(map(_num2words, filter(lambda x: len(x) > 0, text.split())))
    text = ''.join(stemmer.lemmatize(text))
    words = list(map(_ordinal_to_cardinal, filter(lambda x: x not in _stopwords, text.split())))
    bigrams = list(map(lambda x: '\t'.join(x), zip(words[:-1], words[1:])))
    return [words, bigrams]


def _process(x):
    return process(' '.join(x))


def _process_df(df):
    return df.apply(_process, axis=1)


def _update_dicts(x, dict1, dict2):
    dict1.add_documents([x[1]])
    dict2.add_documents([x[2]])
    return 0


def get_dictionaries():
    if not (os.path.exists('dict_1.gensim') and os.path.exists('dict_2.gensim')):
        dictionary_1 = corpora.Dictionary()
        dictionary_2 = corpora.Dictionary()
        pool = Pool(CORE_NUM)
        for i in tqdm(range(math.ceil(NUM_ROWS_TO_PROCESS / ROWS_BUFFER_SIZE))):
            df = pd.read_csv('./docs.tsv', sep='\t', nrows=ROWS_BUFFER_SIZE, skiprows=i * ROWS_BUFFER_SIZE, header=None, index_col=0)
            df = utils.parallelize(pool, df, _process_df, CORE_NUM)
            df.apply(lambda x: _update_dicts(x, dictionary_1, dictionary_2), axis=1)
        dictionary_1.save('dict_1.gensim')
        dictionary_2.save('dict_2.gensim')
    dictionary_1 = corpora.Dictionary.load('dict_1.gensim')
    dictionary_2 = corpora.Dictionary.load('dict_2.gensim')
    return dictionary_1, dictionary_2


if __name__ == '__main__':
    get_dictionaries()

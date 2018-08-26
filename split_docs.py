import os
import sys
import pandas as pd
import math
from tqdm import tqdm
from docs import normalize, get_stopwords, stopwords, stemmer
import multiprocessing
from functools import lru_cache


DOCS_FILE = './docs.tsv'
DOCS_PATH = './docs'
SAMPLE_FILE = './sample.csv'
SPLIT_VALUE = 2000
DOCS_NUM = 582167
CORE_NUM = 8
current_index = None
current_data = None


def get_doc(doc_no):
    global current_data, current_index
    split = doc_no // SPLIT_VALUE
    if split == current_index:
        return current_data.loc[doc_no]
    current_index = split
    current_data = pd.read_csv('./docs/docs_part_%d' % split, sep='\t', index_col=0, header=None)
    return current_data.loc[doc_no]


def loop_body(i):
    print("Starting %d" % i)
    df = pd.read_csv(DOCS_FILE, skiprows=i * SPLIT_VALUE, nrows=SPLIT_VALUE, sep='\t', index_col=0, header=None, dtype={1: str, 2: str})
    df = df.applymap(normalize)
    df.to_csv(os.path.join(DOCS_PATH, 'docs_part_%d' % i), sep='\t', header=None)
    print("Ending %d" % i)


def main():
    get_stopwords()
    if not os.path.exists(DOCS_PATH):
        os.mkdir(DOCS_PATH)
    # docs_in_sample = set(pd.read_csv(SAMPLE_FILE, sep=',', usecols=[1])['DocumentId'].tolist())
    pool = multiprocessing.Pool(CORE_NUM)
    pool.map(loop_body, range(math.ceil(DOCS_NUM / SPLIT_VALUE)))
    '''
    for i in tqdm(range(math.ceil(DOCS_NUM / SPLIT_VALUE))):
        df = pd.read_csv(DOCS_FILE, skiprows=i * SPLIT_VALUE, nrows=SPLIT_VALUE, sep='\t', index_col=0, header=None, dtype={1: str, 2: str})
        df = df.applymap(normalize)
        df.to_csv(os.path.join(DOCS_PATH, 'docs_part_%d' % i), sep='\t', header=None)
        break
    '''
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()

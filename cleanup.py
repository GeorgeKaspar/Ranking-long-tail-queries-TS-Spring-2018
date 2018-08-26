import numpy as np
import csv
from lang import *
from termcolor import colored
import pandas as pd
from docs import normalize, get_stopwords, stopwords, stemmer
from tqdm import tqdm
import jamspell
from google import get_google_spelling


def process(query):
    query = query.replace(',', ' ').replace(':', ' ').replace('/', ' ').replace('\\', ' ').replace('\"', ' ').replace('\'', ' ')
    query = ' '.join(filter(lambda x: len(x) > 0, query.split()))
    print(colored(query, 'red'))
    query = change_keyboard_layout(query)
    print(colored(query, 'green'))
    return [query]


def keyboard_layout_queries():
    with open("queries.tsv", "r") as q_file_from, open("queries_new.tsv", "w") as q_file_to:
        reader = csv.reader(q_file_from, delimiter='\t')
        writer = csv.writer(q_file_to, delimiter='\t')
        for row in tqdm(reader):
            i = int(row[0])
            query = process(row[1])
            writer.writerow([row[0]] + query)


def normalize_queries():
    get_stopwords()
    corrector = jamspell.TSpellCorrector()
    corrector.LoadLangModel('./jamspell_model/ru_small.bin')
    with open("queries_new.tsv", "r") as q_file_from, open("queries_norm.tsv", "w") as q_file_to:
        reader = csv.reader(q_file_from, delimiter='\t')
        writer = csv.writer(q_file_to, delimiter='\t')
        for row in tqdm(reader):
            i = int(row[0])
            query = row[1]
            query = get_google_spelling(query)
            query = normalize(query)
            # query = corrector.FixFragment(row[1])
            writer.writerow([row[0], query])


if __name__ == '__main__':
    normalize_queries()
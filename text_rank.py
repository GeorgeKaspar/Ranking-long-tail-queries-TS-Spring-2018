import numpy as np
import pandas as pd
from gensim.models import doc2vec, TfidfModel
import gc
import os
from tqdm import tqdm
from gensim import corpora, matutils
import logging
from split_docs import get_doc
import csv


def iter_docs_queries():
    df = pd.read_csv('./queries_norm.tsv', sep='\t', header=None, index_col=0)
    for idx, row in tqdm(df.iterrows()):
        yield doc2vec.LabeledSentence(str(row[1]).split(), ['QUERY_%d' % idx])

    for filename in tqdm(os.listdir('./docs/')):
        path = os.path.join('./docs/', filename)
        df = pd.read_csv(path, sep='\t', index_col=0, header=None)
        for idx, row in df.iterrows():
            yield doc2vec.LabeledSentence((str(row[1]) + ' ' + str(row[2])).split(), ['DOC_%d' % idx])


def iter_words():
    d = corpora.Dictionary.load('./dict_1.gensim')
    for filename in tqdm(os.listdir('./docs/')):
        path = os.path.join('./docs/', filename)
        df = pd.read_csv(path, sep='\t', index_col=0, header=None)
        for idx, row in df.iterrows():
            yield d.doc2bow((str(row[1]) + ' ' + str(row[2])).split())


def iter_bigrams():
    d = corpora.Dictionary.load('./dict_2.gensim')
    for filename in tqdm(os.listdir('./docs/')):
        path = os.path.join('./docs/', filename)
        df = pd.read_csv(path, sep='\t', index_col=0, header=None)
        for idx, row in df.iterrows():
            words_title = str(row[1]).split()
            words_content = str(row[2]).split()
            yield d.doc2bow(list(map(lambda x: '\t'.join(x), zip(words_content[:-1], words_content[1:]))) + list(map(lambda x: '\t'.join(x), zip(words_title[:-1], words_title[1:]))))


def train_doc2vec():
    doc2vec_model = doc2vec.Doc2Vec(size=100, window=8, min_count=5, workers=8, iter=30, alpha=1e-2, min_alpha=1e-2)
    doc2vec_model.build_vocab(iter_docs_queries())
    doc2vec_model.train(iter_docs_queries(), total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.iter)
    doc2vec_model.save('doc2vec_weigths')


def train_tfidf():
    tfidf_unigram_model = TfidfModel(iter_words())
    tfidf_unigram_model.save('tfidf_unigram')
    tfidf_bigram_model = TfidfModel(iter_bigrams())
    tfidf_bigram_model.save('tfidf_bigram')


def make_scores_for_sample():
    doc2vec_model = doc2vec.Doc2Vec.load('doc2vec_weigths')
    logging.info('doc2vec loaded')
    tfidf_unigram_model = TfidfModel.load('tfidf_unigram')
    logging.info('tfidf unigram loaded')
    tfidf_bigram_model = TfidfModel.load('tfidf_bigram')
    logging.info('tfidf bigram loaded')
    d1 = corpora.Dictionary.load('./dict_1.gensim')
    logging.info('dict1 loaded')
    d2 = corpora.Dictionary.load('./dict_2.gensim')
    logging.info('dict2 loaded')
    queries = pd.read_csv('./queries_norm.tsv', sep='\t', header=None, index_col=0)
    sample = pd.read_csv('./sample.csv', sep=',').sort_values(by=['DocumentId'])
    with open('./submission.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['QueryId', 'DocumentId', 'Score'])
        for idx, row in tqdm(sample.iterrows()):
            query_id = row['QueryId']
            doc_id = row['DocumentId']
            doc2vec_score = doc2vec_model.docvecs.similarity('DOC_%d' % doc_id, 'QUERY_%d' % query_id)
            doc = get_doc(doc_id)
            query = str(queries.loc[query_id])
            doc_title = str(doc[1])
            doc_content = str(doc[2])

            doc_title_words = doc_title.split()
            doc_content_words = doc_content.split()
            query_words = query.split()

            doc_title_bigrams = d2.doc2bow(list(map(lambda x: '\t'.join(x), zip(doc_title_words[:-1], doc_title_words[1:]))))
            doc_content_bigrams = d2.doc2bow(list(map(lambda x: '\t'.join(x), zip(doc_content_words[:-1], doc_content_words[1:]))))
            query_bigrams = d2.doc2bow(list(map(lambda x: '\t'.join(x), zip(query_words[:-1], query_words[1:]))))

            doc_title_words = d1.doc2bow(doc_title_words)
            doc_content_words = d1.doc2bow(doc_content_words)
            query_words = d1.doc2bow(query_words)

            doc_title_words = tfidf_unigram_model[doc_title_words]
            doc_content_words = tfidf_unigram_model[doc_content_words]
            query_words = tfidf_unigram_model[query_words]

            doc_title_bigrams = tfidf_bigram_model[doc_title_bigrams]
            doc_content_bigrams = tfidf_bigram_model[doc_content_bigrams]
            query_bigrams = tfidf_bigram_model[query_bigrams]

            tfidf_title_score_uni = matutils.cossim(doc_title_words, query_words)
            tfidf_content_score_uni = matutils.cossim(doc_content_words, query_words)
            tfidf_title_score_bi = matutils.cossim(doc_title_bigrams, query_bigrams)
            tfidf_content_score_bi = matutils.cossim(doc_content_bigrams, query_bigrams)

            score = (2 * tfidf_content_score_bi + 2 * tfidf_title_score_uni + tfidf_content_score_uni + 0.5 * doc2vec_score) / 5.5
            writer.writerow([query_id, doc_id, score])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    make_scores_for_sample()

__author__ = 'yangbin1729'

import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = '123@456#789$012%'


class DevelopmentConfig(Config):
    DEBUG = True
    MODEL_DIR = os.path.join(basedir, 'models')
    Word2Vec_DIR = os.path.join(MODEL_DIR,
                                'word2vec\wiki_corpus_above200.model')
    LTP_DATA_DIR = os.path.join(MODEL_DIR, 'ltp')
    CLASSIFIER_DIR = os.path.join(MODEL_DIR, 'classifier')
    TOKENIZER = os.path.join(CLASSIFIER_DIR, 'tokenizer.pickle')

    DATA_DIR = os.path.join(basedir, 'datasets')
    corpus_path = os.path.join(DATA_DIR, 'qa_corpus.csv')
    stopwords_path = os.path.join(DATA_DIR, 'stopword.txt')


class ProductionConfig(Config):
    DEBUG = False
    MODEL_DIR = r'/home/student/project/project-01/noam/project01/models'
    Word2Vec_DIR = os.path.join(MODEL_DIR,
                                'word2vec/wiki_corpus_above200.model')
    LTP_DATA_DIR = os.path.join(MODEL_DIR, 'ltp')
    CLASSIFIER_DIR = r'/home/student/project/project-01/noam/project01' \
                     r'/classifiers'
    TOKENIZER = os.path.join(MODEL_DIR, 'classifier/tokenizer.pickle')

    DATA_DIR = os.path.join(basedir, 'datasets')
    corpus_path = os.path.join(DATA_DIR, 'qa_corpus.csv')
    stopwords_path = os.path.join(DATA_DIR, 'stopword.txt')

config = {'development': DevelopmentConfig, 'production': ProductionConfig, }
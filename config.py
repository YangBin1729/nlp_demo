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


class ProductionConfig(Config):
    DEBUG = False
    MODEL_DIR = r'/home/student/project/project-01/noam/project01/models'
    Word2Vec_DIR = os.path.join(MODEL_DIR,
                                'word2vec/wiki_corpus_above200.model')
    LTP_DATA_DIR = os.path.join(MODEL_DIR, 'ltp')
    CLASSIFIER_DIR = os.path.join(MODEL_DIR, 'classifier')
    TOKENIZER = os.path.join(CLASSIFIER_DIR, 'tokenizer.pickle')


config = {'development': DevelopmentConfig, 'production': ProductionConfig, }
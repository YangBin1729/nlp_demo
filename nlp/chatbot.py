import os
from collections import defaultdict
from functools import reduce
from itertools import combinations
import jieba
import pandas as pd
# from scipy.spatial.distance import cosine
# from sklearn.feature_extraction.text import TfidfVectorizer
from config import config

__author__ = 'yangbin1729'

corpus_path = config['development'].corpus_path
stopwords_path = config['development'].stopwords_path


def union(set_list):
    return reduce(lambda x, y: x & y, set_list)


def in_k_set(set_list, k):
    if k >= len(set_list):
        result = union(set_list)
    else:
        result = set()
        for s in combinations(set_list, k):
            temp = union(s)
            result.update(temp)
    return result


class ChatBot:
    def __init__(self):
        corpus = pd.read_csv(corpus_path, encoding='utf-8')
        self.corpus = corpus

        inv_table = defaultdict(set)
        self.stopwords = [w.strip() for w in
                          open(stopwords_path, encoding='utf-8').readlines()]
        for ind, row in enumerate(corpus['question']):
            word_list = jieba.cut(str(row))
            for w in word_list:
                if w not in self.stopwords:
                    inv_table[w].add(ind)
        self.inv_table = inv_table

        # vectorizer = TfidfVectorizer(max_features=10000)
        # vectors = vectorizer.fit_transform(
        #     corpus['question'].apply(lambda t: ' '.join(jieba.cut(str(t)))))
        # self.vectorizer = vectorizer
        # self.vectors = vectors.toarray()

    def get_response(self, input_text):
        input_words = jieba.cut(input_text.strip())
        # input_vec = \
        # self.vectorizer.transform([' '.join(input_words)]).toarray()[0]
        candidate = self.get_candidate(input_words)
        if not candidate:
            return "未找到答案，请重新输入问题。"
        response_id = self.filter(candidate)
        return self.corpus['answer'].iloc[response_id]

    def get_candidate(self, input_words):
        set_list = [self.inv_table[w] for w in input_words if
                    w not in self.stopwords]

        candidate = set()
        for k in range(len(set_list), 0, -1):
            temp = in_k_set(set_list, k)
            candidate.update({(k, i) for i in temp})
            if len(candidate) >= 10:
                return candidate
        return candidate

    def filter(self, candidate):
        # todo:布尔搜索后如何筛选结果
        sorted_candidate = sorted(candidate, key=lambda s: -s[0])
        return sorted_candidate[0][1]


# bot = ChatBot()
# print(bot.get_response('密码修改'))

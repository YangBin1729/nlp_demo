__author__ = 'yangbin1729'

import os
import numpy as np
from collections import defaultdict
from pyltp import SentenceSplitter, Segmentor, Postagger, NamedEntityRecognizer, \
    Parser
from config import config

# from key_words import model

path = os.path.abspath(os.path.dirname(__file__))
key_words_path = os.path.join(path, 'key_words.txt')

LTP_DATA_DIR = config['production'].LTP_DATA_DIR
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')

# 每次都是 load ，然后 release ，可以写个支持 with 语句的类
segmentor = Segmentor()
segmentor.load(cws_model_path)

postagger = Postagger()
postagger.load(pos_model_path)

recognizer = NamedEntityRecognizer()
recognizer.load(ner_model_path)

parser = Parser()
parser.load(par_model_path)

with open(key_words_path, 'r', encoding='utf-8') as f:
    key_words = f.read().split(' ')


def news_parser(news):
    sents = SentenceSplitter.split(news)
    sents = [sent.strip() for sent in sents if sent]

    who_what = []
    for sent in sents:
        who, what = sent_parser(sent)
        who_what.append((who, what))

    # for i, (who, what) in enumerate(who_what):
    #     if who and what:
    #         expect = 0.7
    #         if cosine_dis(sen_vec(what), sen_vec(sents[i+1])) > expect:
    #             what += sents[i+1]
    #             who_what[i] = (who_what)

    return [(who, what) for (who, what) in who_what if who and what]


# def sen_vec(sent):
#     words = list(segmentor.segment(sent))
#     sen_vec = np.sum([model[k] for k in words], axis=1) / len(words)
#     return sen_vec


def cosine_dis(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * (np.linalg.norm(vec2)))


def sent_parser(sent):
    """
    三种简单的形式，标点符号是关键：
    —— 王二说，今天天气很好。
    —— 王二说：“今天天气很好。”
    —— 今天天气很好，王二说。
    todo:更复杂点的句型
    """
    words = segmentor.segment(sent)
    postags = postagger.postag(words)
    netags = recognizer.recognize(words, postags)
    arcs = parser.parse(words, postags)

    parse_list = build_parse_list(words, arcs)

    who, what = '', ''

    for i in range(len(words)):
        if words[i] in key_words and postags[i] == 'v':
            parse_dict = parse_list[i]

            if 'SBV' in parse_dict:
                index_of_who = parse_dict['SBV'][0]
                if 'Nh' in netags[index_of_who] or 'Ni' in netags[index_of_who]:
                    who = words[index_of_who]

                if ('VOB' in parse_dict) or ('COO' in parse_dict):
                    index_of_comma = [j for j in range(len(words[i:]))
                                      if words[i:][j] in
                                      (',', '，', ':', '：')][0]
                    what = ''.join(words[i + 1 + index_of_comma:])

                if arcs[i].relation == 'COO':
                    index_of_comma = \
                    [j for j in range(len(words[:index_of_who]))
                     if words[j] in (',', '，')][-1]
                    what = ''.join(words[:index_of_comma])

    return who, what


# todo:利用树形结构来保存依存关系，然后解析说了什么


def build_parse_list(words, arcs):
    """
    找出和每个单词有依存关系的词典，词典项为(关系：单词列表)

    >>> words = ['元芳', '你', '怎么', '看']
    >>> print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        4:'SBV', 4:'SBV', 4:'ADV', 0:'HED'
    >>> build_parse_list(words, arcs)
        [ {}, {}, {}, {'SBV':[0,1], 'ADV':[2]} ]

    """
    parse_list = []
    for i in range(len(words)):
        child_dict = defaultdict(list)
        for j, arc in enumerate(arcs):
            if arc.head == i + 1:
                child_dict[arc.relation].append(j)
        # if 'SBV' in child_dict:
        #    print(words[i],child_dict['SBV'])
        parse_list.append(child_dict)
    return parse_list


def release_model():
    segmentor.release()
    postagger.release()
    recognizer.release()
    parser.release()


class OpinionExtractor:
    def __init__(self):
        self.splitter = SentenceSplitter

        self.segmentor = Segmentor()
        self.segmentor.load(cws_model_path)

        self.postagger = Postagger()
        self.postagger.load(pos_model_path)

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(ner_model_path)

        self.parser = Parser()
        self.parser.load(par_model_path)

        with open(key_words_path, 'r', encoding='utf-8') as f:
            self.key_words = f.read().split(' ')

    def extractor(self, texts):
        sents = self.splitter.split(texts)
        results = []
        for i, sent in enumerate(sents):
            p, o = self.sent_parser(sent)
            if p and o:
                results.append((p, o))
        return results

    def sent_parser(self, sent):
        words = self.segmentor.segment(sent)
        postags = self.postagger.postag(words)
        netags = self.recognizer.recognize(words, postags)
        arcs = self.parser.parse(words, postags)

        for i, w in enumerate(words):
            if w in self.key_words:
                pass
            return w


if __name__ == '__main__':
    text1 = '小明对他说，今天天气很好'
    sent1 = '天气真好，适合打球'
    text2 = '黄涛家人称，步道离河很近，设置了栅栏，但为什么会开一道门，' \
            '而且没上锁，正是因为这一疏忽，导致了这场悲剧。'
    text3 = '今天天气很好，适合打球，高二三班的李克强说。'

    doc = '刚刚在记者会上，针对个别西方媒体传出的辞职传言，中国香港特首林郑月娥说：“我留意到一个私人秘密的聚会里面交流被公开，我认为非常不适当，很失望。我在两个多月前已经回答过，从开始到现在，我都从未向中央提过辞职，原因是我有信心带领团队帮香港走出困局，所以并不存在我想辞职结果辞不了的这种矛盾。我作为香港的行政长官，经历了一些心路历程，个人情绪有很大的波动，但最终的决定是为香港市民考虑，找到一个共同的方向和目标。我和我的团队非常努力在创造这个目标，但需要大家的合作。”'
    print(sent_parser(text1))
    print(sent_parser(text2))
    print(sent_parser(text3))
    print(news_parser(doc))
    release_model()

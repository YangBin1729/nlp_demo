__author__ = 'yangbin1729'

import os
from gensim.models import word2vec
from pyltp import Postagger


key_words_path = 'key_words.txt'
LTP_DATA_DIR = r'D:\Program\ltp_data_v3.4.0'
model_path = r"D:\Program\wiki\wiki_corpus_above200.model"

pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')

model = word2vec.Word2Vec.load(model_path)
postagger = Postagger()
postagger.load(pos_model_path)


# 说 的所有同义词
def get_key_words(initial_word):

    key_words = []
    frontier = [(initial_word, 1.0)]
    explored = []

    while frontier:
        word, similarity = frontier.pop()
        if word not in explored:
            if word not in key_words and similarity > 0.5 and \
                    model.wv.similarity(initial_word, word) > 0.4:
                key_words.append(word)
            explored.append(word)

            most_similar = model.wv.most_similar(word)
            most_similar_verbs = get_verbs(most_similar)

            frontier.extend(most_similar_verbs)
            frontier.sort(key=lambda s: s[1])
    return key_words


def get_verbs(most_similar):
    words = [word for word, similarity in most_similar]
    tags = postagger.postag(words)
    return [most_similar[i] for i in range(len(most_similar)) if tags[i]=='v']


if __name__ == '__main__':
    initial_word = '说'
    key_words = get_key_words(initial_word)
    postagger.release()
    with open(key_words_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(key_words))



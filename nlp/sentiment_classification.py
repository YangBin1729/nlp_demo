import json
import os
import pickle
import re
import string

import requests
import zhconv
import jieba
from keras.preprocessing import sequence
from config import config


class Result:
    label2zh = {'location_traffic_convenience': '交通是否便利',
                'location_distance_from_business_district': '距离商圈远近',
                'location_easy_to_find': '是否容易寻找',
                'service_wait_time': '排队等候时间',
                'service_waiters_attitude': '服务人员态度',
                'service_parking_convenience': '是否容易停车',
                'service_serving_speed': '点菜/上菜速度', 'price_level': '价格水平',
                'price_cost_effective': '性价比', 'price_discount': '折扣力度',
                'environment_decoration': '装修情况', 'environment_noise': '嘈杂情况',
                'environment_space': '就餐空间', 'environment_cleaness': '卫生情况',
                'dish_portion': '分量', 'dish_taste': '口感', 'dish_look': '外观',
                'dish_recommendation': '推荐程度',
                'others_overall_experience': '本次消费感受',
                'others_willing_to_consume_again': '再次消费的意愿', 'location': '位置',
                'service': '服务', 'price': '价格', 'environment': '环境',
                'dish': '菜品', 'others': '其它'}
    
    label_layers = {'location': ['location_traffic_convenience',
                                 'location_distance_from_business_district',
                                 'location_easy_to_find'],
                    'service': ['service_wait_time', 'service_waiters_attitude',
                                'service_parking_convenience',
                                'service_serving_speed'],
                    'price': ['price_level', 'price_cost_effective',
                              'price_discount'],
                    'environment': ['environment_decoration',
                                    'environment_noise', 'environment_space',
                                    'environment_cleaness'],
                    'dish': ['dish_portion', 'dish_taste', 'dish_look',
                             'dish_recommendation'],
                    'others': ['others_overall_experience',
                               'others_willing_to_consume_again']}
    
    sentiment = ['未提及', '负向', '中性', '正向']
    
    def render(self, prob_dict, mode='tree'):
        if mode == 'tree':
            results = {'name': '点评', 'children': []}
            for layer1 in self.label_layers.keys():
                layer1_zh = self.label2zh[layer1]
                layer1_dict = {'name': layer1_zh, 'children': []}
                for layer2 in self.label_layers[layer1]:
                    probs = prob_dict.get(
                        layer2,{"predictions": [[0, 0, 0, 0]]})["predictions"][0]
                    label = ['未提及', '负向', '中性', '正向']
                    layer2_zh = self.label2zh[layer2]
                    prob_with_label = [{'name': l, 'value': str(round(v, 2))}
                                       for l, v in zip(label, probs)]
                    # 如何可视化这结果？？？
                    layer2_dict = {'name': layer2_zh,
                                   'children': prob_with_label}
                    layer1_dict['children'].append(layer2_dict)
                results['children'].append(layer1_dict)
            return results


class SentimentClassifier:
    def __init__(self):
        clfs_dir = config['production'].CLASSIFIER_DIR
        clfs_path = {p: os.path.join(clfs_dir, p) for p in os.listdir(clfs_dir)
                     if os.path.isdir(os.path.join(clfs_dir, p))}
        base_api = 'http://localhost:8012/v1/models/{}:predict'
        clfs_api = {cate: base_api.format(cate) for cate in clfs_path}
    
        with open(config['production'].TOKENIZER, 'rb') as f:
            tokenizer = pickle.load(f)
    
        self.tokenizer = tokenizer
        self.clfs_api = clfs_api
    
    def preprocess(self, text):
        punct = r"，。！？、；：“”\n＂＃＄％＆＇（）＊＋－／＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､" \
                r"〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〟〰〾〿–—‛„‟…‧﹏★☆•→▽"
        
        def clean_special_chars(text):
            text = text.encode('utf-8').decode('utf-8')
            re_tok = re.compile(f'([{string.punctuation}{punct}])')
            return re_tok.sub(r' ', text)
        
        def simplify(text):
            return zhconv.convert(text, 'zh-cn')
        
        def cut_join(text):
            space = ' '
            words = jieba.cut(text)
            return space.join([w.lower() for w in words if not w.isspace()])
        
        sents = re.split('[.!?。！？]', text.strip())
        results = []
        for sent in sents:
            if sent:
                sent = clean_special_chars(sent)
                sent = simplify(sent)
                sent = cut_join(sent)
                results.append(sent)
        return ' '.join(results)

    def text2vector(self, text):
        text = self.preprocess(text)
        text_seq = self.tokenizer.texts_to_sequences([text])
        text_seq = sequence.pad_sequences(text_seq, maxlen=250)
        return text_seq

    def predict(self, text):
        results = {}
        X_input = self.text2vector(text)
    
        X_input = X_input.astype('int').tolist()[0]
        print(X_input)
        payload = {"instances": [{'input_text': X_input}]}
        for cate in self.clfs_api.keys():
            r = self.single_predict(cate, payload)
            results[cate] = r
        return self.format_probs(results)
    
    def single_predict(self, cate, payload):
        api = self.clfs_api[cate]
        r = requests.post(api, json=payload)
        if r.ok:
            print(cate, r.content.decode('utf-8'))
            result = json.loads(r.content.decode('utf-8'))
        else:
            result = {"predictions": [[0.0, 0.0, 0.0, 0.0]]}
        return result
    
    def format_probs(self, probs):
        r = Result()
        results = r.render(probs, mode='tree')
        
        return results
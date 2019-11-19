# import glob
# import os
# import pickle
# import re
#
# import tensorflow as tf
# from keras.models import load_model
#
# __author__ = 'yangbin1729'
#
# # current_path = os.path.dirname(__file__)
# # tokenizer_path = os.path.join(current_path, 'tokenizer.pickle')
#
#
# def load_tokenizer():
#     with open(tokenizer_path, 'rb') as f:
#         tokenizer = pickle.load(f)
#     return tokenizer
#
#
# def load_classifiers():
#     model_names = glob.glob(f'{current_path}\*.h5')
#     print(model_names)
#     classifiers = {}
#     for m in model_names[:1]:
#         label_name = re.findall(r'.+\\(\w+)\.', m)[-1]
#         print(f"Load model for  <{label_name}> classification...")
#         model = load_model(m)
#         model._make_predict_function()
#         print('-'*50)
#         classifiers[label_name] = model
#     return classifiers

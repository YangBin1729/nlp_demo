__author__ = 'yangbin1729'

import os
import re

import tensorflow as tf
import glob

tf.keras.backend.set_learning_phase(0)

model_dir = '../models/classifier/'
models_path = glob.glob((f"{model_dir}/*.h5"))
export_dir = '../classifiers'
with tf.keras.backend.get_session() as sess:
    for path in models_path[:2]:
        model = tf.keras.models.load_model(path)
        name = re.findall(r'.+/(\w+)\.h5', path)[0]
        export_path = os.path.join(export_dir, name + '/1')
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_text':model.input},
            outputs={t.name: t for t in model.outputs}
        )
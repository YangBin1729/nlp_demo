__author__ = 'yangbin1729'

import numpy as np
import requests

text = np.zeros(250)
for i in range(100, 250):
    text[i] = np.random.randint(1, 10000)

text = text.astype('int').tolist()

payload = {"instances": [{'input_text': text}]}

r1 = requests.post('http://localhost:8012/v1/models/dish_look:predict',
                   json=payload)

r2 = requests.post('http://localhost:8012/v1/models/dish_portion:predict',
                   json=payload)

print('location_traffic_convenience:', r1.content.decode('utf-8'))
print('service_serving_speed:',r2.content.decode('utf-8'))


# docker run --name yangbin1729 -p 8012:8501 --mount type=bind,source=$(pwd)/classifiers,target=/models --mount type=bind,source=$(pwd)/classifiers/models.config,target=/models/models.config -t tensorflow/serving --model_config_file=/models/models.config
import numpy as np
import os
from model import model, class_names
from keras.utils import img_to_array
from pdf2image import convert_from_path

model.load_weights('classifier_weights.weights.h5')
if not os.path.isdir('Workspace/solution'):
    os.mkdir('Workspace/solution')
for i in doc_class:
    if not os.path.isdir(f'Workspace/{i}'):
        os.mkdir(f'Workcpace/{i}')
for j in os.listdir('Workspace/pdf/'):
    data = convert_from_path(f'Workspace/pdf/{j}')
    for i in range(len(data)):
        pred = model.predict(img_to_array(data[i]))
        doc_class = class_names[np.argmax(pred)]
        data[i].save(f'Workspace/solution/{doc_class}/{doc_class}{j}{i}.jpg', 'JPEG')

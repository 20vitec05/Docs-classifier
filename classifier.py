import numpy as np
import os, fitz
from model import model, class_names
from keras.utils import img_to_array
from PIL import Image

model.load_weights('classifier_weights.weights.h5')
if not os.path.isdir('Workspace/solution'):
    os.mkdir('Workspace/solution')
for i in class_names:
    if not os.path.isdir(f'Workspace/solution/{i}'):
        os.mkdir(f'Workspace/solution/{i}')
for j in os.listdir('Workspace/pdf/'):
    data = fitz.open(f'Workspace/pdf/{j}')
    for i in range(len(data)):
        page = data.load_page(i)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image = image.resize((400,400))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        doc_class = class_names[np.argmax(pred)]
        image.save(f'Workspace/solution/{doc_class}/{doc_class}{j}{i}.jpg', 'JPEG')

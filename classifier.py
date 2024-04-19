import numpy as np
from model import model, class_names
from keras.utils import img_to_array
from pdf2image import convert_from_path

model.load_weights('classifier_weights.weights.h5')
data = convert_from_path('Workspace/pdf/example.pdf')
for i in range(len(data)):
    pred = model.predict(img_to_array(data[i]))
    data[i].save(class_names[np.argmax(pred)] + str(i) +'.jpg', 'JPEG')

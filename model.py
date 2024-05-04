import pathlib

import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, RandomFlip, RandomRotation, Lambda
import numpy as np
from transformers import ViTImageProcessor, TFAutoModelForImageClassification
from sklearn.utils.class_weight import compute_class_weight


train_dir = pathlib.Path('Dataset/train')
test_dir = pathlib.Path('Dataset/test')

img_width, img_height = 400, 400
input_shape = (img_width, img_height, 3)

batch_size = 4
epochs= 30

train_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    seed=2452,
    batch_size=batch_size,
    image_size=(img_height, img_width),
    subset='training',
    label_mode='categorical'
)

val_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    seed=2452,
    batch_size=batch_size,
    image_size=(img_height, img_width),
    subset='validation',
    label_mode='categorical'
)

test_dataset = image_dataset_from_directory(
    test_dir,
    batch_size=batch_size,
    image_size=(img_height, img_width),
    label_mode='categorical'
)

class_names = train_dataset.class_names
num_classes = len(class_names)
labels = []
for images, labels_batch in train_dataset:
    labels.extend(np.argmax(labels_batch, axis=1))
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))



model_name = "google/vit-base-patch16-224"
# processor = ViTImageProcessor.from_pretrained(model_name)
model = TFAutoModelForImageClassification.from_pretrained(model_name)

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(train_dataset, validation_data= val_dataset, epochs=10)
model.evaluate(test_dataset)



# base_model = TFAutoModelForImageClassification.from_pretrained(model_name)

# data_augm = Sequential([
#     RandomFlip('horizontal'),
#     RandomRotation(0.2)
# ])

# fine_tuning = Sequential([
#     Dense(512, activation='relu'),
#     Dropout(0.25),
#     Dense(256, activation='relu',kernel_regularizer='l2'),
#     Dense(128, activation='relu',kernel_regularizer='l2'),
#     Dense(num_classes, activation='softmax')
# ])

# input = Input(shape=input_shape)
# x = data_augm(input)
# x = processor(images=x, return_tensors="tf")
# x = base_model(x, training=False)['logits']
# output = fine_tuning(x)

# model = Model(input, output)

# model.compile(optimizer='SGD',
#             loss='categorical_crossentropy',
#             metrics=['accuracy'])

# model.fit(train_dataset, validation_data= val_dataset, epochs= epochs)
# model.save_weights('classifier_weights.weights.h5')
# model.evaluate(test_dataset)

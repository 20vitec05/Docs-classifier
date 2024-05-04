import pathlib

import numpy as np
from transformers import ViTFeatureExtractor, TFAutoModelForImageClassification
from keras._tf_keras.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
from keras._tf_keras.keras.models import Sequential, Model
from keras._tf_keras.keras.layers import Input, Dense, Dropout, Flatten, RandomFlip, RandomRotation
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.applications.resnet50 import preprocess_input


train_dir = pathlib.Path('ImageDataset/train')
test_dir = pathlib.Path('ImageDataset/test')

img_width, img_height = 224, 224
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


model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
base_model = TFAutoModelForImageClassification.from_pretrained(model_name)

data_augm = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2)
])

fine_tuning = Sequential([
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(256, activation='relu',kernel_regularizer='l2'),
    Dense(128, activation='relu',kernel_regularizer='l2'),
    Dense(num_classes, activation='softmax')
])

input = Input(shape=input_shape)
x = data_augm(input)
x = base_model(x)
output = fine_tuning(x)

model = Model(input, output)

model.compile(optimizer='SGD',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
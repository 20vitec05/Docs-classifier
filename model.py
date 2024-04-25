import pathlib
import tensorflow as tf
from keras._tf_keras.keras.utils import image_dataset_from_directory
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras._tf_keras.keras.applications import ResNet50

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dir = pathlib.Path('Dataset/train')
test_dir = pathlib.Path('Dataset/test')

img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)

batch_size = 20
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

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

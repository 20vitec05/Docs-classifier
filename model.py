import pathlib
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout, Activation
train_dir = pathlib.Path('Dataset/train')
test_dir = pathlib.Path('Dataset/test')

img_width, img_height = 400, 400
input_shape = (img_width, img_height, 3)

epochs = 30
batch_size = 20
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

model = Sequential([
    Conv2D(32, (3,3), padding='same',activation='relu', input_shape = input_shape),
    MaxPooling2D((2,2),strides=2),

    Conv2D(64, (3,3), padding='same',activation='relu'),
    MaxPooling2D((2,2),strides=2),

    Flatten(),

    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(num_classes,activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, validation_data= val_dataset, epochs= epochs)
model.save_weights('classifier_weights.weights.h5')
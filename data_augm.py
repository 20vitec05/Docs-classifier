import os
import matplotlib.pyplot as plt
from keras._tf_keras.keras.preprocessing import image

if not os.path.exists('Dataset/newtrain'):
    os.makedirs('Dataset/newtrain')
for i in os.listdir('Dataset/idk'):
    if not os.path.exists(f'Dataset/newtrain/{i}'):
        os.makedirs(f'Dataset/newtrain/{i}')
augs = [
    image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=155,
    ),
    image.ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.5,
    ),
    image.ImageDataGenerator(
        rescale=1./255,
        height_shift_range=.25,
    ),
    image.ImageDataGenerator(
        rescale=1./255,
        brightness_range=(.1, .5),
    ),
    image.ImageDataGenerator(
        rescale=1./255,
        shear_range=90
    ),
    image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True
    ),
    image.ImageDataGenerator(
        rescale=1./255,
        vertical_flip=True
    )
]
for i in os.listdir('Dataset/idk'):
    for j in range(len(os.listdir(f'Dataset/idk/{i}'))):
        for augmentation in augs:   
            generator = augmentation.flow_from_directory(
                directory='Dataset/idk', 
                batch_size=1,
                save_to_dir=f'Dataset/newtrain/{i}',
                save_prefix='idk',
                save_format='png'
            )
            next(generator)
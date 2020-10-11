import os
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import CategoricalCrossentropy

import numpy as np
import matplotlib.pyplot as plt

base_dir = 'rockpaperscissors1'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')

train_paper_dir = os.path.join(train_dir, 'paper')
train_rock_dir = os.path.join(train_dir, 'rock')
train_scissor_dir = os.path.join(train_dir, 'scissors')

validation_paper_dir = os.path.join(validation_dir, 'paper')
validation_rock_dir = os.path.join(validation_dir, 'rock')
validation_scissor_dir = os.path.join(validation_dir, 'scissors')

train_datagen = image.ImageDataGenerator(
  rescale=1./255,
  rotation_range=0.2,
  zoom_range=0.2,
  width_shift_range=0.2,
  vertical_flip=True,
  horizontal_flip=True,
  validation_split= 0.4
)

test_datagen = image.ImageDataGenerator(
  rescale=1./255,
  rotation_range=0.2,
  zoom_range=0.2,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  vertical_flip=True,
  horizontal_flip=True,
  validation_split=0.4,
)

train_generator = train_datagen.flow_from_directory(
  train_dir,
  # batch_size=400,
  target_size=(100, 150),)

validation_generator = test_datagen.flow_from_directory(
  validation_dir,
  # batch_size=160,
  target_size=(100, 150),)

model = Sequential([
  layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 150, 3)),
  layers.MaxPooling2D(2, 2),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Conv2D(128, (3,3), activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Conv2D(256, (3,3), activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  # layers.Dense(128, activation='relu'),
  # layers.Dense(64, activation='relu'),
  # layers.Dense(32, activation='relu'),
  layers.Dense(3, activation='softmax')
])

model.compile(loss=CategoricalCrossentropy(),
  optimizer= Adadelta(),
  metrics=['accuracy'])

model.fit(
  train_generator,
  validation_data=validation_generator,
  steps_per_epoch=30,
  validation_steps=12,
  epochs=20,
  verbose=1)

img1 = image.load_img("rockpaperscissors/test/paper/papertest.jpg", target_size=(100, 150))

x1 = image.img_to_array(img1)
x1 = np.expand_dims(x1, axis=0)

images1 = np.vstack([x1])
classes = model.predict(images1, batch_size=20)

print(classes)

img2 = image.load_img("rockpaperscissors/test/rock/rocktest.jpg", target_size=(100, 150))

x2 = image.img_to_array(img2)
x2 = np.expand_dims(x2, axis=0)

images2 = np.vstack([x2])
classes = model.predict(images2, batch_size=20)

print(classes)
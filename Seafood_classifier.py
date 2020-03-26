#import statements
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt

#dataset
data = (r"C:\Users\aacjp\Desktop\data\Seafood")

base_dir = (r"C:\Users\aacjp\Desktop\data\Seafood")

#split into train and test
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

#split into class names for train and test
train_Salmon_dir = os.path.join(train_dir, 'Salmon')
train_Crab_dir = os.path.join(train_dir, 'Crab')
train_Lobster_Tails_dir = os.path.join(train_dir, 'Lobster_Tails')
train_Mussels_dir = os.path.join(train_dir, 'Mussels')
train_Oysters_dir = os.path.join(train_dir, 'Oysters')
train_Sea_Scallops_dir = os.path.join(train_dir, 'Sea_Scallops')
train_Shrimp_dir = os.path.join(train_dir, 'Shrimp')
train_Swordfish_dir = os.path.join(train_dir, 'Swordfish')
train_Yellowfin_Tuna_dir = os.path.join(train_dir, 'Yellowfin_Tuna')

validation_Salmon_dir = os.path.join(validation_dir, 'Salmon')
validation_Crab_dir = os.path.join(validation_dir, 'Crab')
validation_Lobster_Tails_dir = os.path.join(validation_dir, 'Lobster_Tails')
validation_Mussels_dir = os.path.join(validation_dir, 'Mussels')
validation_Oysters_dir = os.path.join(validation_dir, 'Oysters')
validation_Sea_Scallops_dir = os.path.join(validation_dir, 'Sea_Scallops')
validation_Shrimp_dir = os.path.join(validation_dir, 'Shrimp')
validation_Swordfish_dir = os.path.join(validation_dir, 'Swordfish')
validation_Yellowfin_Tuna_dir = os.path.join(validation_dir, 'Yellowfin_Tuna')

num_Salmon_tr = len(os.listdir(train_Salmon_dir))
num_Crab_tr = len(os.listdir(train_Crab_dir))
num_Lobster_Tails_tr = len(os.listdir(train_Lobster_Tails_dir))
num_Mussels_tr = len(os.listdir(train_Mussels_dir))
num_Oysters_tr = len(os.listdir(train_Oysters_dir))
num_Sea_Scallops_tr = len(os.listdir(train_Sea_Scallops_dir))
num_Shrimp_tr = len(os.listdir(train_Shrimp_dir))
num_Swordfish_tr = len(os.listdir(train_Swordfish_dir))
num_Yellowfin_Tuna_tr = len(os.listdir(train_Yellowfin_Tuna_dir))

num_Salmon_val = len(os.listdir(validation_Salmon_dir))
num_Crab_val = len(os.listdir(validation_Crab_dir))
num_Lobster_Tails_val = len(os.listdir(validation_Lobster_Tails_dir))
num_Mussels_val = len(os.listdir(validation_Mussels_dir))
num_Oysters_val = len(os.listdir(validation_Oysters_dir))
num_Sea_Scallops_val = len(os.listdir(validation_Sea_Scallops_dir))
num_Shrimp_val = len(os.listdir(validation_Shrimp_dir))
num_Swordfish_val = len(os.listdir(validation_Swordfish_dir))
num_Yellowfin_Tuna_val = len(os.listdir(validation_Yellowfin_Tuna_dir))

total_train = num_Salmon_tr +  num_Crab_tr + num_Lobster_Tails_tr + num_Mussels_tr + num_Oysters_tr + num_Sea_Scallops_tr + num_Shrimp_tr + num_Swordfish_tr + num_Yellowfin_Tuna_tr 
total_val = num_Salmon_val + num_Crab_val + num_Lobster_Tails_val + num_Mussels_val + num_Oysters_val + num_Sea_Scallops_val + num_Shrimp_val + num_Swordfish_val + num_Yellowfin_Tuna_val

#print("Total training images:", total_train)
#print("Total validation images:", total_val)

batch_size = total_val
epochs = 20
IMG_HEIGHT = 150
IMG_WIDTH = 150

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

class_names = ['Salmon', 'Crab', 'Lobster Tails', 'Mussels', 'Oysters', 'Sea Scallops', 'Shrimp', 'Swordfish', 'Tilapia', 'Yellowfin_Tuna']

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

sample_training_images, _ = next(train_data_gen)
sample_testing_images, _ = next(val_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
#plotImages(sample_training_images[:5])

#build model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), input_shape=(150, 150, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(15, activation='softmax'))
#model.summary()


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#save model so it can be used multiple times without retraining
model.save("model.h5")
#model = keras.models.load_model("model.h5")

prediction = model.predict(sample_testing_images)

plt.imshow(sample_testing_images[1053])
plt.show()

print(class_names[np.argmax(prediction[1053])])
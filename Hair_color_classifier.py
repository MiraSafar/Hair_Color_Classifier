# Tensorflow hair color classifier trained on Kindgirls nude models' faces

# Import libraries
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import csv
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# This fixed the cudNN error: "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize."
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# This should limit the messages from Tensorflow using the imported "os" library
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

dataset_labels = [1, 2, 3, 4] # 1 = black, 2 = blond, 3 = brown, 4 = red

dataset_folder = "C:/Users/miros/faces/augmented_images"
dataset_labels_file = "C:/Users/miros/faces/girlnames.csv"

# Loading labels and image filenames from the CSV file
csv_label_file = open(dataset_labels_file, "r")

reader = csv.reader(csv_label_file)

girls = []

for row in reader:
    girls.append(row)

# Split it into 2 lists 
dataset_images, dataset_labels = map(list, zip(*girls))


# Convert the dataset labels from string to integers. For example "brown" to 3, "blond" to 2 etc.
# This is needed for the Tensorflow functions to work properly
for n,i in enumerate(dataset_labels):
    if i == 'black':
        dataset_labels[n] = 1
    if i == 'blond':
        dataset_labels[n] = 2
    if i == 'brown':
        dataset_labels[n] = 3
    if i == 'red':
        dataset_labels[n] = 4

#
#
# From Tensorflow manual
#
#


# Define parameters for the loader:
batch_size = 32
img_height = 200
img_width = 200

# Use Keras method to load dataset from directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_folder,         # Please note that the path to the folder has to be one above the actual folder containing the images :(
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

val_ds = tf.keras.preprocessing.image_dataset_from_directory(      # Using the 'subset' tag with 'training' and 'validation' splits the original singular data into two groups automatically
    dataset_folder,         # Please note that the path to the folder has to be one above the actual folder containing the images :(
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
    )

# Show the image batch structure and the labels structure:
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

class_names = train_ds.class_names
print(class_names)

# Import Tensorflow layers
from tensorflow.keras import layers

# Create a model
num_classes = 4

# Create a data augmentation layer which will be added to the model in a variable "data_augmentation"
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.2),
  layers.experimental.preprocessing.RandomContrast(0.2),
  layers.experimental.preprocessing.RandomZoom(0.2),
])

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255), # The RGB channel values are in the [0, 255] range. This is not ideal for a neural network; in general you should seek to make your input values small. Here, we will standardize values to be in the [0, 1] by using a Rescaling layer.
    data_augmentation, # This data augmentation could also be written out directly into this function as layers.experimental.preprocessing."anyAugmentationFunction"
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
    )

#
#
# Test model accuracy on never seen testing data
#
#

# Open directory with testing data and load a sample image into a variable
test_image = PIL.Image.open('C:/Users/miros/Projects/Hair color classification/testing_data_augmented/3.jpg')

# See if the loaded testing image has the right size and format:
print(test_image.format)
print(test_image.size)
print(test_image.mode)

# Save sample image into an Numpy array - this is needed for Tensorflow
data = np.asarray(test_image)

print(data.shape)

# Create an array of arrays of the numpy array with testing image - this is needed because our model expects the input data in batch (different shape than supplying with just one dimensional array)
sample_to_predict = np.array([data])
print(sample_to_predict.shape)

# Make a hair color prediction of our sample image:
predictions = model.predict(sample_to_predict)
print(predictions)  # Discrete probability distribution will be returned for each model class

print(class_names[np.argmax(predictions)])  # Select the class with the highest probability (returns a single number) and load that class name from "class_names"




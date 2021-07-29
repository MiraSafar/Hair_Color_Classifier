# Image resizer

# This code is rubish!!! Not suitable for publication.

# However, it serves its purpose, so I will keep it.
    
# Import libraries
from PIL import Image
import os
import numpy as np

# Check if all files in the dataset folder are jpg

folder = 'C:/Users/miros/Datasets/faces/original_images/images'
augmented_dataset_folder = 'C:/Users/miros/Datasets/faces/300x300_images/'

image_type_jpg = 0
image_type_other = 0
    
for filename in os.listdir(folder):
    if filename.endswith('jpg'):
        image_type_jpg = image_type_jpg + 1
    else:
        image_type_other = image_type_other +1

print("There are {} jpg images".format(image_type_jpg))
print("There are {} other images".format(image_type_other))


# Check for all used formats in the original images

count_200x200 = 0
count_400x400 = 0
count_210x210 = 0
count_300x300 = 0
count_other = 0

for filename in os.listdir(folder):
    image = Image.open(folder + "/" + filename)
    if image.size == (200, 200):
        count_200x200 = count_200x200 + 1
    elif image.size == (400, 400):
        count_400x400 = count_400x400 + 1
    elif image.size == (210, 210):
        count_210x210 = count_210x210 + 1
    elif image.size == (300, 300):
        count_300x300 = count_300x300 + 1
    else:
        count_other = count_other + 1
        print(image.size)
        print(filename)

# Resize all images to 200x200 dimensions and save them to a new augmented dataset folder
for filename in os.listdir(folder):
    image = Image.open(folder + "/" + filename)
    print(image)
    new_image = image.resize((300, 300))
    new_image.save(augmented_dataset_folder + filename)


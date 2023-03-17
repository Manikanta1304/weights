import pickle
from PIL import Image
import os

# Define the directory path containing the images
img_dir = "/path/to/image/directory"

# Create an empty list to store the images
img_list = []

# Loop over the images in the directory
for img_file in os.listdir(img_dir):
    # Load the image file using PIL
    img = Image.open(os.path.join(img_dir, img_file))
    # Append the image to the list
    img_list.append(img)

# Define the filename to save the pickled data to
output_file = "image_data.pkl"

# Pickle the list of images and save to file
with open(output_file, "wb") as f:
    pickle.dump(img_list, f)


import pickle

# Define the filename containing the pickled data
input_file = "image_data.pkl"

# Load the pickled data from file
with open(input_file, "rb") as f:
    img_list = pickle.load(f)

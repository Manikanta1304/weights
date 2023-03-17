import h5py
import os
from PIL import Image
import numpy as np

# set up the path to the directory containing the images
image_dir = 'path/to/directory'

# set up the path to the output HDF5 file
hdf5_file = 'images.hdf5'

# create a new HDF5 file
with h5py.File(hdf5_file, 'w') as hf:

    # iterate over all the image files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):

            # load the image file into memory
            path = os.path.join(image_dir, filename)
            with Image.open(path) as img:
                img_arr = np.array(img)

            # create a new dataset in the HDF5 file
            ds = hf.create_dataset(filename, data=img_arr)


import h5py
from PIL import Image
import numpy as np

# set up the path to the HDF5 file
hdf5_file = 'images.hdf5'

# open the HDF5 file
with h5py.File(hdf5_file, 'r') as hf:

    # iterate over all the datasets in the file
    for name in hf:

        # read the dataset into memory
        img_arr = hf[name][()]

        # create a PIL image from the array
        img = Image.fromarray(img_arr)

        # display the image
        img.show()

import zipfile
import os

# set up the path to the directory containing the images
image_dir = 'path/to/directory'

# set up the path to the output ZIP archive file
archive_file = 'images.zip'

# create a new ZIP archive file
with zipfile.ZipFile(archive_file, 'w') as zipf:

    # iterate over all the image files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            
            # add the file to the archive
            path = os.path.join(image_dir, filename)
            zipf.write(path, arcname=filename)


from PIL import Image
import zipfile
import io

# set up the path to the ZIP archive file
archive_file = 'images.zip'

# open the archive file
with zipfile.ZipFile(archive_file, 'r') as zipf:

    # iterate over all the files in the archive
    for filename in zipf.namelist():
        if filename.endswith('.jpg') or filename.endswith('.png'):

            # read the file contents into memory
            with zipf.open(filename) as f:
                image_bytes = io.BytesIO(f.read())

            # load the image from the bytes
            with Image.open(image_bytes) as img:
                img.show()

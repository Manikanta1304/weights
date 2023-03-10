import streamlit as st
import os
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title='Image Explorer', page_icon=':camera:')

# Set up sidebar
st.sidebar.title('Image Explorer')
directory = st.sidebar.text_input('Enter directory path:', './images-dir')

# Get image paths
image_extensions = ['.jpg', '.jpeg', '.png']
image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(tuple(image_extensions))]

# Define number of columns in the grid
cols = st.sidebar.slider('Number of columns:', 2, 6, 4)

# Define size of the images in the grid
image_size = 200

# Create a container for the image grid
container = st.container()

# Create a container for the selected image
selected_image_container = st.container()

# Calculate number of rows in the grid
rows = len(image_paths) // cols + 1

# Display the image grid in the container
with container:
    # Display the title and a horizontal line
    st.write('# Image Grid')
    st.write('---')

    # Create the grid columns
    grid = [st.columns(cols) for i in range(rows)]

    # Iterate over the images and display each one in the grid
    for index, image_path in enumerate(image_paths):
        if index < len(image_paths):
            image = Image.open(image_path)
            image.thumbnail((image_size, image_size))
            j = index % cols
            i = index // cols
            grid[i][j].image(image, use_column_width=True, caption=os.path.basename(image_path))
            if grid[i][j].button('Select', key=f'select_button_{index}'):
                selected_image = Image.open(image_path)
                selected_image.thumbnail((600, 600))
                selected_image_container.empty()
                selected_image_container.image(selected_image, use_column_width=True, caption=os.path.basename(image_path))

# Set the container height to display a scrollbar when there are many images
container_height = str(rows * 210) + 'px'  # 210px is the height of each row
container._st_container.height = container_height

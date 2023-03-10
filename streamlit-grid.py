# import os
# import streamlit as st
# from PIL import Image

# def get_image_paths(directory):
    # image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    # image_paths = []
    # for root, dirs, files in os.walk(directory):
        # for filename in files:
            # if filename.endswith(image_extensions):
                # image_paths.append(os.path.join(root, filename))
    # return image_paths

# st.sidebar.title('Image Explorer')
# directory = st.sidebar.text_input('Directory', 'images-dir')
# image_paths = get_image_paths(directory)

# if len(image_paths) == 0:
    # st.warning('No images found in directory')
# else:
    # selected_image_path = st.selectbox('Select an image', image_paths)
    # image = Image.open(selected_image_path)
    # st.image(image, use_column_width=True)


import os
import streamlit as st
from PIL import Image

def get_image_paths(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(image_extensions):
                image_paths.append(os.path.join(root, filename))
    return image_paths

st.sidebar.title('Image Explorer')
directory = st.sidebar.text_input('Directory', 'images-dir')
image_paths = get_image_paths(directory)

if len(image_paths) == 0:
    st.warning('No images found in directory')
else:
    columns = st.sidebar.slider('Number of columns', 1, 6, 3)
    selected_image_path = st.sidebar.empty()
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        images.append(image)

    rows = int(len(images) / columns) + (1 if len(images) % columns > 0 else 0)
    for i in range(rows):
        cols = st.columns(columns)
        for j in range(columns):
            index = i * columns + j
            if index < len(images):
                cols[j].image(images[index], use_column_width=True, caption=os.path.basename(image_paths[index]))
                if cols[j].button('Select', key=f'select_button_{index}'):
                    selected_image_path.image(images[index], use_column_width=True, caption=os.path.basename(image_paths[index]))


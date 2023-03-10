import os
import streamlit as st
from PIL import Image

# Set the path to your directory of images
path = 'images-dir'

# Get a list of all the image files in the directory
image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]

# Create a dropdown menu with the list of image files in the sidebar
selected_image = st.sidebar.selectbox('Select an image', image_files)

# Load the selected image using PIL and display it in Streamlit
if selected_image:
    image_path = os.path.join(path, selected_image)
    image = Image.open(image_path)
    st.image(image, caption=selected_image, use_column_width=True)



# import os
# from azure.storage.blob import BlobServiceClient

# # Set up the connection to your Azure Storage account
# connection_string = "DefaultEndpointsProtocol=https;AccountName=<account_name>;AccountKey=<account_key>;EndpointSuffix=core.windows.net"
# blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# # Set the name of the container and the directory inside the container where the images are stored
# container_name = "<container_name>"
# directory_name = "<directory_name>"

# # Get a list of all the blobs (i.e., files) in the specified directory in the container
# blobs = blob_service_client.list_blobs(container_name, prefix=directory_name)

# # Extract the names of the image files from the list of blobs
# image_files = [blob.name.split('/')[-1] for blob in blobs if blob.name.endswith('.jpg')]

# # Create a dropdown menu with the list of image files in the sidebar
# selected_image = st.sidebar.selectbox('Select an image', image_files)

# # Load the selected image from the Azure Storage container and display it in Streamlit
# if selected_image:
    # blob_client = blob_service_client.get_blob_client(container_name=container_name, blob_name=os.path.join(directory_name, selected_image))
    # image_bytes = blob_client.download_blob().readall()
    # st.image(image_bytes, caption=selected_image, use_column_width=True)

import os
import pandas as pd
import streamlit as st
from PIL import Image
 
st.set_page_config(layout="wide")

st.sidebar.title('Image Explorer')
# Get the path to the directory containing the images
image_directory = st.sidebar.text_input('Enter path to image directory:', '../images-dir')

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
        
st.markdown(f'''
    <style>
    section[data-testid="stSidebar"] .css-ng1t4o {{width: 14rem;}}
    </style>
''',unsafe_allow_html=True)



# Define CSS styles for the vertical line
line_styles = """
  border-left: 3px solid white;
  height: 800px;
  position: absolute;
  left: 20%;
  right: 0%;
  margin-left: -1px;
  top: 0;
"""

# Read the CSV file with the rim types for each image
try:
    group_file = None #st.sidebar.file_uploader('Upload group CSV file:', type=['csv'])
    if group_file is None:
        group_df = pd.read_csv('image-groups.csv')
    else:
        group_df = pd.DataFrame(columns=['image_filename', 'Rim Type'])
 
    # Group the images by rim type and display the grid
    grouped = group_df.groupby('Rim Type')
    col1, col2, col3, = st.columns([1, 0.05, 2])
    # with col1:
        # st.write('# Rim Type')
        # st.write('---')
    rim_type = st.sidebar.selectbox('Select Rim Type:', options=['All'] + list(grouped.groups.keys()))
 
    with col1:
        # # Display the title and a horizontal line
        # st.write('# Image Explorer')
        # if rim_type == 'All':
            # st.write('### Group', rim_type)
            # filtered_image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory)]
        # else:
            # filtered_image_paths = [os.path.join(image_directory, filename) for filename in group_df[group_df['Rim Type'] == rim_type]['image_filename']]
        # st.write('---')
        
        # rows = len(filtered_image_paths) // 4 + 1
        # # Create the grid columns
        # grid = [st.columns(3) for i in range(rows)]

        # # Iterate over the images and display each one in the grid
        # for index, image_path in enumerate(filtered_image_paths): #[start_index:start_index+20]):
            # if index < 20:
                # image = Image.open(image_path)
                # image.thumbnail((200, 200))
                # j = index % 3
                # i = index // 3
                # grid[i][j].image(image, use_column_width=True) #, use_column_width=True, caption=os.path.basename(image_path))
                # # Display image and select button
                # button_key = f"select_button_{index}"
                # if grid[i][j].button(label="✅", key=button_key, help=f"Select {image_path}"):
                    # selected_image_path = image_path
                    
        st.write('# Images')
        st.write('---')
        if rim_type == 'All':
            image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory)]
        else:
            image_paths = [os.path.join(image_directory, filename) for filename in group_df[group_df['Rim Type'] == rim_type]['image_filename']]
        selected_image_path = st.selectbox('Select an image:', options=[''] + image_paths)
    
    
    with col2:
        # # st.empty()
        st.markdown(f'<div style="{line_styles}">', unsafe_allow_html=True)
        
    with col3:
        st.write('# Full Size Image')
        st.write('---')
        if selected_image_path:
            st.image(selected_image_path)
            
            # Get the filename of the selected image
            selected_image_filename = os.path.basename(selected_image_path)
            
            # Get the current rim type for the selected image from the CSV file
            current_rim_type = group_df[group_df['image_filename'] == selected_image_filename]['Rim Type'].iloc[0]
            
            # # Create a form to update the rim type for the selected image
            # if 'new_rim_type' not in st.session_state:
                # st.session_state.new_rim_type = current_rim_type
                
            with st.form(key='update_rim_type_form'):
                st.text_input('Update Rim Type:', value=current_rim_type, key='new_rim_type')
                submit_button = st.form_submit_button(label='Update')
                
            # If the form is submitted, update the rim type in the CSV file
            if submit_button:
                group_df.loc[group_df['image_filename'] == selected_image_filename, 'Rim Type'] = st.session_state.new_rim_type
                # group_df = group_df.groupby('Rim Type').apply(lambda x: x.reset_index(drop=True))
                group_df.to_csv('image-groups.csv', index=False)
                st.success('Rim Type updated successfully!')
                # Refresh the page to update the image grid with the updated group information
                st.experimental_rerun()
 
        else:
            st.write("Select an image from the grid to view.")
 
except FileNotFoundError as e:
    st.write('Invalid Directory', e)

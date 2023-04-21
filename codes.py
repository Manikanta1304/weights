import Levenshtein

strings = ["apple", "banana", "cherry", "aple", "bananana"]

for i in range(len(strings)):
    for j in range(i + 1, len(strings)):
        distance = Levenshtein.distance(strings[i], strings[j])
        print(f"Levenshtein distance between '{strings[i]}' and '{strings[j]}' is {distance}")


import streamlit as st
import pandas as pd
import tempfile
import os

@st.experimental_singleton
def get_predictions(image_dir):
    results_df = run(image_dir)
    return results_df
    
sidebar_col, grid_col, detect_col = st.columns([1,2,3])

with sidebar_col:
    uploaded_files = st.file_uploader("Upload image", accept_multiple_files=True, type=['png', 'jpeg', 'jpg', 'JPG'], key='image_dir')
    if uploaded_files is not None:
        # Create a temporary directory to store the uploaded files
        temp_dir = tempfile.TemporaryDirectory()
        for uploaded_file in uploaded_files:
            # Save each uploaded file to the temporary directory
            uploaded_file_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(uploaded_file_path, 'wb') as f:
                f.write(uploaded_file.read())
        # Use the temporary directory as the image directory for YOLO
        image_dir = temp_dir.name
        results_df = get_predictions(image_dir)
        st.experimental_singleton.set('df', results_df)
        

if st.experimental_singleton.get('df') is not None:
    results_df = st.experimental_singleton.get('df')
    gb = GridOptionsBuilder.from_dataframe(results_df)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gridOptions = gb.build()
    data = AgGrid(results_df,
                  gridOptions=gridOptions, 
                  enable_enterprise_modules=True, 
                  allow_unsafe_jscode=True, 
                  update_mode=GridUpdateMode.SELECTION_CHANGED)
    
    if len(data["selected_rows"])>0:
        detected_img = Image.open(os.path.join(image_dir, data["selected_rows"][0]['file_name']))
        st.image(detected_img, channels='BGR')










def get_predictions():
    results_df = run(st.session_state['image_dir'])
    st.session_state['df'] = results_df
    
    
sidebar_col, grid_col, detect_col = st.columns([1,2,3])

with sidebar_col:
    uploaded_files = st.file_uploader("Upload image", accept_multiple_files=True, type=['png', 'jpeg', 'jpg', 'JPG'], key='image_dir')
    if uploaded_files is not None:
        # Create a temporary directory to store the uploaded files
        temp_dir = tempfile.TemporaryDirectory()
        uploaded_file_paths = []
        for uploaded_file in uploaded_files:
            # Save each uploaded file to the temporary directory and store its path
            uploaded_file_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(uploaded_file_path, 'wb') as f:
                f.write(uploaded_file.read())
            uploaded_file_paths.append(uploaded_file_path)
        # Use the first uploaded file as the image directory for YOLO
        st.session_state['image_dir'] = uploaded_file_paths[0]


# You can then use the 'get_predictions' function to run the object detection model and display the results as before.



def get_predictions():
    results_df = run(st.session_state['image_dir'])
    st.session_state['df'] = results_df
    
    
sidebar_col, grid_col, detect_col = st.columns([1,2,3])

with sidebar_col:
    directory = st.selectbox('Select directory path:', ('', './test-images',), on_change=get_predictions, key='image_dir')

import streamlit as st
import pandas as pd
import os
from PIL import Image
from streamlit_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode

def run(data_mount_path):
    # code for object detection
    # saves the results to a csv file './predictions.csv'

st.title("Dot Detector")
st.write("You can view real-time object detection done using YOLO model here.")
st.markdown(
f'''
<style>
.sidebar .sidebar-content {{width:250}}
.css-zbg2rx {{width:13rem !important}}   
</style>
''',
unsafe_allow_html = True)

directory = st.sidebar.text_input('Enter directory path:', './test-images')

if st.button('Detect'):
    run(data_mount_path=directory)
    results_df = pd.read_csv('./predictions.csv')

    if not results_df.empty:
        gb = GridOptionsBuilder.from_dataframe(results_df)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        gridOptions = gb.build()
        data = AgGrid(results_df,
                      gridOptions=gridOptions, 
                      enable_enterprise_modules=True, 
                      allow_unsafe_jscode=True, 
                      update_mode=GridUpdateMode.VALUE_CHANGED)
        
        # Store the selected rows in a Streamlit variable
        selected_rows = data["selected_rows"]
        st.session_state.selected_rows = selected_rows

if 'selected_rows' in st.session_state and st.session_state.selected_rows:
    selected_rows = st.session_state.selected_rows
    for row in selected_rows:
        detected_img = Image.open(os.path.join('./test_images', row['file_name']))
        st.image(detected_img, channels='BGR')


	
	
	
	
st.title("Dot Detector")
st.write("You can view real-time object detection done using YOLO model here.")
st.markdown(
f'''
<style>
.sidebar .sidebar-content {{width:250}}
.css-zbg2rx {{width:13rem !important}}   
</style>
''',
unsafe_allow_html = True)

# uploaded_file = st.sidebar.file_uploader("Upload image", accept_multiple_files=True, type=['png', 'jpeg', 'jpg', 'JPG'])
directory = st.sidebar.text_input('Enter directory path:', './test-images')

if st.button('Detect'):
    run(data_mount_path = directory)


    results_df = pd.read_csv('./predictions.csv')
    gb = GridOptionsBuilder.from_dataframe(results_df)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gridOptions = gb.build()
    data = AgGrid(results_df,
                  gridOptions=gridOptions, 
                  enable_enterprise_modules=True, 
                  allow_unsafe_jscode=True, 
                  update_mode=GridUpdateMode.SELECTION_CHANGED)
    
    if len(data["selected_rows"])>0:
        detected_img = Image.open(os.path.join('./test_images', data["selected_rows"][0]['file_name']))
        st.image(detected_img, channels='BGR')





uploaded_file = st.sidebar.file_uploader("Upload image", accept_multiple_files=True, type=['png', 'jpeg', 'jpg', 'JPG'])

if uploaded_file:
if len(uploaded_file) == 1:
    uploaded_file = uploaded_file[0]
    with st.spinner(text='loading...'):
	picture = Image.open(uploaded_file)
	# picture = picture.resize((5184,3888))
	st.sidebar.image(picture)
	score_threshold = st.sidebar.slider("Confidence_threshold", 0.00,1.00,0.5,0.01)
	nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)
	picture1 = picture.save(f'../yolov5_files/input/{uploaded_file.name}')
	print('picture saved')


    if st.button('detect'):
	source = os.path.sep.join([os.getcwd(), f'../yolov5_files/input/{uploaded_file.name}'])
	results_df = run(weights=f"../best_yolo_v5_reannotate_32.pt", save_txt=f"{uploaded_file.name.split('.')[0]}.txt", 
		    conf_thres=score_threshold, source=source, save_conf=True)

	directory = os.path.sep.join([os.getcwd(),'../yolov5_files/runs/detect'])
	runs_print = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
	detected_img = Image.open(os.path.join(runs_print, uploaded_file.name))

	gb = GridOptionsBuilder.from_dataframe(results_df)
	gb.configure_pagination()
	gb.configure_side_bar()
	gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
	gb.configure_selection(selection_mode="multiple", use_checkbox=True)
	gridOptions = gb.build()
	data = AgGrid(results_df.head(1), 
		      gridOptions=gridOptions, 
		      enable_enterprise_modules=True, 
		      allow_unsafe_jscode=True, 
		      update_mode=GridUpdateMode.SELECTION_CHANGED)
		
import os
import shutil

# set the source and destination folders
source_folders = ['/path/to/folder1', '/path/to/folder2', '/path/to/folder3']
destination_folder = '/path/to/destination/folder'

# iterate through the source folders
for folder in source_folders:
    # get a list of all the files in the folder
    files = os.listdir(folder)
    # iterate through the files and move them to the destination folder
    for file_name in files:
        full_file_name = os.path.join(folder, file_name)
        if os.path.isfile(full_file_name):
            shutil.move(full_file_name, destination_folder)

	
	
import streamlit as st
import pandas as pd
   
    
df = pd.read_csv('tmp.csv')
img_id = st.number_input("img_id",1,4,1)
rim_type = st.text_input("rim type", df[df.img_id == img_id]['rim_type'].values[0])

if st.button('add'): 
    df.loc[df.img_id==img_id,'rim_type'] = rim_type
    df.to_csv('tmp.csv',index=False)
st.dataframe(df)


import streamlit as st
import os
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")
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
  height: 1300px;
  position: absolute;
  left: 20%;
  right: 0%;
  margin-left: -1px;
  top: 0;
"""
    
    
st.sidebar.title('Image Explorer')
directory = st.sidebar.text_input('Enter directory path:', '../images-dir')
group_file = st.sidebar.file_uploader("Upload image-groups.csv", type=["csv"])

# Set up the layout of the page with two columns
col1, col2, col3, = st.columns([0.8, 0.05, 1])
# col1, col3, = st.columns([0.7, 1], gap='small')

# Initialize the index of the first image to be displayed
start_index = st.session_state.get('start_index', 0)

# Get image paths
image_extensions = ['.jpg', '.jpeg', '.png']

try:
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(tuple(image_extensions))] 
 
    # Load the group information from the CSV file
    if group_file:
        group_df = pd.read_csv(group_file)
        # Get the unique group labels from the CSV file
        group_labels = ['All'] + list(group_df['Rim Type'].unique())

        # Set up the group selection dropdown
        selected_group = st.sidebar.selectbox('Select group', group_labels)

        # Filter the images by the selected group
        if selected_group == 'All':
            filtered_image_paths = image_paths
        else:
            filtered_image_filenames = group_df[group_df['Rim Type'] == selected_group]['image_filename']
            filtered_image_paths = [os.path.join(directory, f) for f in filtered_image_filenames]
    else:
        selected_group = ''
        filtered_image_paths = image_paths

    # Calculate number of rows in the grid
    rows = len(filtered_image_paths) // 4 + 1

    selected_image_path = ''

    # Display the image grid in the container
    with col1:
        # Display the title and a horizontal line
        st.write('# Image Explorer')
        if selected_group:
            st.write('### Group', selected_group)
        st.write('---')

        # Create the grid columns
        grid = [st.columns(3) for i in range(rows)]

        # Iterate over the images and display each one in the grid
        for index, image_path in enumerate(filtered_image_paths[start_index:start_index+20]):
            if index < 20:
                image = Image.open(image_path)
                image.thumbnail((200, 200))
                j = index % 3
                i = index // 3
                grid[i][j].image(image, use_column_width=True) #, use_column_width=True, caption=os.path.basename(image_path))
                # Display image and select button
                button_key = f"select_button_{index}"
                if grid[i][j].button(label="Select", key=button_key, help=f"Select {image_path}"):
                    print(image_path.replace('thumbnails', 'original'))
                    image_path = image_path.replace('thumbnails', 'original')
                    selected_image_path = image_path
                    
                
    prev_col, next_col = st.sidebar.columns([1,1])

    with prev_col:
        # Add a button to load the previous set of images
        if start_index > 0:
            if st.button('Previous'):
                # Update the index of the first image to be displayed
                start_index -= 20
                st.session_state['start_index'] = start_index
                # Refresh the page to load the previous set of images
                st.experimental_rerun()

    with next_col:
        # Add a button to load the next set of images
        if start_index + 20 < len(filtered_image_paths):
            if st.button('Next', key='next_button'):
                # Update the index of the first image to be displayed
                start_index += 20
                st.session_state['start_index'] = start_index
                # Refresh the page to load the next set of images
                st.experimental_rerun()
                
               

               
    with col2:
        # # st.empty()
        st.markdown(f'<div style="{line_styles}">', unsafe_allow_html=True)
        

    with col3:
        # st.markdown(f'<div style="{line_styles}">', unsafe_allow_html=True)
        st.write('# Full Size Image')
        st.write('---')
        if selected_image_path:
            st.image(selected_image_path)
            # Get the filename of the selected image
            selected_image_filename = os.path.basename(selected_image_path)
            # Get the current rim type for the selected image from the CSV file
            current_rim_type = group_df[group_df['image_filename'] == selected_image_filename]['Rim Type'].iloc[0]
            # Create a form to update the rim type for the selected image
            with st.form(key='update_rim_type_form'):
                new_rim_type = st.text_input('Update Rim Type:', value=current_rim_type)
                submit_button = st.form_submit_button(label='Update')
            # If the form is submitted, update the rim type in the CSV file
            if submit_button:
                group_df.loc[group_df['image_filename'] == selected_image_filename, 'Rim Type'] = new_rim_type
                group_df = group_df.groupby('Rim Type').apply(lambda x: x.reset_index(drop=True))
                group_df.to_csv(group_file.name, index=False)
                st.success('Rim Type updated successfully!')
                # Refresh the page to update the image grid with the updated group information
                st.experimental_rerun()
        else:
            st.write('sfdsf')
            st.write("Select an image from the grid to view.")

    
except FileNotFoundError as e:
    st.write('Invalid Directory', e)






import streamlit as st
import os
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")
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
col1_style = """
  left: 20%;
  right: 0%;
  margin-left: -1px;
  top: 0;
"""


# Define CSS styles for the vertical line
line_styles = """
  border-left: 3px solid white;
  height: 1300px;
  position: absolute;
  left: 20%;
  right: 0%;
  margin-left: -1px;
  top: 0;
"""
    
    
st.sidebar.title('Image Explorer')
directory = st.sidebar.text_input('Enter directory path:', './dot-3k-thumbnails')
group_file = st.sidebar.file_uploader("Upload image-groups.csv", type=["csv"])

# Set up the layout of the page with two columns
col1, col2, col3, = st.columns([0.8, 0.05, 1], gap='small')
# col1, col3, = st.columns([0.7, 1], gap='small')

# Initialize the index of the first image to be displayed
start_index = st.session_state.get('start_index', 0)

# Get image paths
image_extensions = ['.jpg', '.jpeg', '.png']

try:
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(tuple(image_extensions))] 
 
    # Load the group information from the CSV file
    if group_file:
        group_df = pd.read_csv(group_file)
        # Get the unique group labels from the CSV file
        group_labels = ['All'] + list(group_df['Rim Type'].unique())

        # Set up the group selection dropdown
        selected_group = st.sidebar.selectbox('Select group', group_labels)

        # Filter the images by the selected group
        if selected_group == 'All':
            filtered_image_paths = image_paths
        else:
            filtered_image_filenames = group_df[group_df['Rim Type'] == selected_group]['image_filename']
            filtered_image_paths = [os.path.join(directory, f) for f in filtered_image_filenames]
    else:
        selected_group = ''
        filtered_image_paths = image_paths

    # Calculate number of rows in the grid
    rows = len(filtered_image_paths) // 4 + 1

    selected_image_path = ''

    # Display the image grid in the container
    with col1:
        # Display the title and a horizontal line
        st.write('# Image Explorer')
        if selected_group:
            st.write('### Group', selected_group)
        st.write('---')

        # Create the grid columns
        grid = [st.columns(3) for i in range(rows)]

        # Iterate over the images and display each one in the grid
        for index, image_path in enumerate(filtered_image_paths[start_index:start_index+20]):
            if index < 20:
                image = Image.open(image_path)
                image.thumbnail((200, 200))
                j = index % 3
                i = index // 3
                grid[i][j].image(image, use_column_width=True) #, use_column_width=True, caption=os.path.basename(image_path))
                # Display image and select button
                button_key = f"select_button_{index}"
                if grid[i][j].button(label="Select", key=button_key, help=f"Select {image_path}"):
                    print(image_path.replace('thumbnails', 'original'))
                    image_path = image_path.replace('thumbnails', 'original')
                    selected_image_path = image_path
                    
                
    prev_col, next_col = st.sidebar.columns([1,1])

    with prev_col:
        # Add a button to load the previous set of images
        if start_index > 0:
            if st.button('Previous'):
                # Update the index of the first image to be displayed
                start_index -= 20
                st.session_state['start_index'] = start_index
                # Refresh the page to load the previous set of images
                st.experimental_rerun()

    with next_col:
        # Add a button to load the next set of images
        if start_index + 20 < len(filtered_image_paths):
            if st.button('Next', key='next_button'):
                # Update the index of the first image to be displayed
                start_index += 20
                st.session_state['start_index'] = start_index
                # Refresh the page to load the next set of images
                st.experimental_rerun()
                
               

               
    with col2:
        # # st.empty()
        st.markdown(f'<div style="{line_styles}">', unsafe_allow_html=True)
        

    with col3:
        # st.markdown(f'<div style="{line_styles}">', unsafe_allow_html=True)
        st.write('# Full Size Image')
        st.write('---')
        if selected_image_path:
            st.image(selected_image_path)
        else:
            st.write("Select an image from the grid to view.")
            
except FileNotFoundError:
    st.write('Invalid Directory')





import os
import random
import pandas as pd

# Set the path to your image directory
image_dir = 'images-dir'

# Get a list of all the image filenames in the directory
image_filenames = os.listdir(image_dir)

# Shuffle the list of image filenames randomly
random.shuffle(image_filenames)

# Define the number of groups
num_groups = 3

# Calculate the number of images per group, rounded to the nearest integer
images_per_group = round(len(image_filenames) / num_groups)

# Create a list of group labels with random integers between 1 and the number of groups
group_labels = [random.randint(1, num_groups) for _ in range(len(image_filenames))]

# Split the group labels into chunks of the same size
group_label_chunks = [group_labels[i:i+images_per_group] for i in range(0, len(group_labels), images_per_group)]

# If the last chunk is smaller than the others, merge it with the previous chunk
if len(group_label_chunks[-1]) < images_per_group // 2:
    group_label_chunks[-2] += group_label_chunks[-1]
    group_label_chunks.pop()

# Flatten the list of group label chunks to create a list of final group labels
group_labels = [label for chunk in group_label_chunks for label in chunk]

# Create a Pandas DataFrame with the image filenames and group labels as columns
df = pd.DataFrame({'image_filename': image_filenames, 'group_label': group_labels})

# Print the first few rows of the DataFrame to check it looks correct
df.group_label.value_counts()




import streamlit as st
import os
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")

st.sidebar.title('Image Explorer')
directory = st.sidebar.text_input('Enter directory path:', './images-dir')
group_file = st.sidebar.file_uploader("Upload image-groups.csv", type=["csv"])

# Set up the layout of the page with two columns
col1, col2, col3, = st.columns([1, 0.2, 0.7])

# Initialize the index of the first image to be displayed
start_index = st.session_state.get('start_index', 0)

# Get image paths
image_extensions = ['.jpg', '.jpeg', '.png']

try:
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(tuple(image_extensions))] 
 
    # Load the group information from the CSV file
    if group_file:
        group_df = pd.read_csv(group_file)
        # Get the unique group labels from the CSV file
        group_labels = ['All'] + sorted(list(group_df['group_label'].unique()))

        # Set up the group selection dropdown
        selected_group = st.sidebar.selectbox('Select group', group_labels)

        # Filter the images by the selected group
        if selected_group == 'All':
            filtered_image_paths = image_paths
        else:
            filtered_image_filenames = group_df[group_df['group_label'] == selected_group]['image_filename']
            filtered_image_paths = [os.path.join(directory, f) for f in filtered_image_filenames]
    else:
        selected_group = ''
        filtered_image_paths = image_paths

    # Calculate number of rows in the grid
    rows = len(filtered_image_paths) // 4 + 1

    selected_image_path = ''

    # Display the image grid in the container
    with col1:
        # Display the title and a horizontal line
        st.write('# Image Explorer')
        if selected_group:
            st.write('## Group', selected_group)
        st.write('---')

        # Create the grid columns
        grid = [st.columns(4) for i in range(rows)]

        # Iterate over the images and display each one in the grid
        for index, image_path in enumerate(filtered_image_paths[start_index:start_index+20]):
            if index < 20:
                image = Image.open(image_path)
                image.thumbnail((200, 200))
                j = index % 4
                i = index // 4
                grid[i][j].image(image, use_column_width=True, caption=os.path.basename(image_path))
                # Display image and select button
                button_key = f"select_button_{index}"
                if grid[i][j].button(label="Select", key=button_key, help=f"Select {image_path}"):
                    selected_image_path = image_path
                    
                
    prev_col, next_col = st.sidebar.columns([1,1])

    with prev_col:
        # Add a button to load the previous set of images
        if start_index > 0:
            if st.button('Previous'):
                # Update the index of the first image to be displayed
                start_index -= 20
                st.session_state['start_index'] = start_index
                # Refresh the page to load the previous set of images
                st.experimental_rerun()

    with next_col:
        # Add a button to load the next set of images
        if start_index + 20 < len(filtered_image_paths):
            if st.button('Next', key='next_button'):
                # Update the index of the first image to be displayed
                start_index += 20
                st.session_state['start_index'] = start_index
                # Refresh the page to load the next set of images
                st.experimental_rerun()
                
               
    # Define CSS styles for the vertical line
    line_styles = """
      border-left: 3px solid white;
      height: 2000px;
      position: absolute;
      left: 50%;
      margin-left: -3px;
      top: 0;
    """
               
    with col2:
        # st.empty()
        st.markdown(f'<div style="{line_styles}">', unsafe_allow_html=True)

    with col3:
        st.write('# Full Size Image')
        st.write('---')
        if selected_image_path:
            st.image(selected_image_path)
        else:
            st.write("Select an image from the grid to view.")
            
except FileNotFoundError:
    st.write('Invalid Directory')








create a script that will scan a folder in CDL and create corresp. folder containing thumbnails under thumbs. this script should run on the VM and take the CDL folder path as an argument

write a streamlit app that

has a drop-down to select the CDL path/directory with images

allows a user to upload a img-groups.csv which contains a single group label for each image (i.e. two columns in the csv - image_filename and group_label)

displays the image thumbnails in groups (as specified by the group_label in the csv)

when a thumbnail is clicked, download the full resolution image from CDL and display on the right



# Fill pixel values greater than 100 with random grayscale values
mask = target_img > 100
gray = np.random.randint(0, 256, size=target_img.shape[:2], dtype=np.uint8)
gray = np.expand_dims(gray, axis=2)
fill = np.tile(gray, [1, 1, 3])
target_img = np.where(mask, fill, target_img)



# Threshold the source image to remove the brightest pixels
source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
source_thresh = cv2.threshold(source_gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
source_thresh = cv2.cvtColor(source_thresh, cv2.COLOR_GRAY2BGR)
source_img_thresh = cv2.bitwise_and(source_img, source_thresh)



# Smooth the target image with a median filter or Gaussian blur
smoothed_img = cv2.medianBlur(target_img, 5)
# smoothed_img = cv2.GaussianBlur(target_img, (5, 5), 0)

# Calculate the mean and standard deviation of the color channels in each image
source_mean, source_std = cv2.meanStdDev(source_img)
target_mean, target_std = cv2.meanStdDev(smoothed_img)





# Smooth the target image with a median filter or Gaussian blur
smoothed_img = cv2.medianBlur(target_img, 5)
# smoothed_img = cv2.GaussianBlur(target_img, (5, 5), 0)


import cv2
import numpy as np

# Load the target image and the source image
target_img = cv2.imread('target_image.jpg')
source_img = cv2.imread('source_image.jpg')

# Resize the source image to match the size of the target image
source_img = cv2.resize(source_img, (target_img.shape[1], target_img.shape[0]))

# Convert the images to grayscale
target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

# Create a mask by thresholding the target image
ret, mask = cv2.threshold(target_gray, 10, 255, cv2.THRESH_BINARY)

# Invert the mask
mask_inv = cv2.bitwise_not(mask)

# Apply the mask to the source image
source_bg = cv2.bitwise_and(source_img, source_img, mask=mask_inv)

# Apply the mask to the target image
target_fg = cv2.bitwise_and(target_img, target_img, mask=mask)

# Blend the source and target images using the mask
result = cv2.add(source_bg, target_fg)

# Show the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()




import cv2
import numpy as np

# Load the target image and the source image
target_img = cv2.imread('target_image.jpg')
source_img = cv2.imread('source_image.jpg')

# Resize the source image to match the size of the target image
source_img = cv2.resize(source_img, (target_img.shape[1], target_img.shape[0]))

# Convert the images to grayscale
target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

# Threshold the target image to create a binary mask
ret, mask = cv2.threshold(target_gray, 10, 255, cv2.THRESH_BINARY)

# Define the kernel for the morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Erode the mask to remove small gaps
mask_eroded = cv2.erode(mask, kernel)

# Dilate the mask to fill in any remaining gaps
mask_dilated = cv2.dilate(mask_eroded, kernel)

# Apply the mask to the source image
result = cv2.bitwise_and(source_img, source_img, mask=mask_dilated)

# Show the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()






import cv2
import numpy as np

# Load the source and target images
source_img = cv2.imread("source.jpg")
target_img = cv2.imread("target.jpg")

# Calculate the mean and standard deviation of the color channels in each image
source_mean, source_std = cv2.meanStdDev(source_img)
target_mean, target_std = cv2.meanStdDev(target_img)

# Normalize the color channels of the target image using the source image statistics
for i in range(3):  # Loop over color channels (B, G, R)
    target_img[:, :, i] = ((target_img[:, :, i] - target_mean[i]) * (source_std[i] / target_std[i])) + source_mean[i]

# Clip the pixel values to the range [0, 255]
target_img = np.clip(target_img, 0, 255).astype(np.uint8)

# Save the resulting image
cv2.imwrite("result.jpg", target_img)







import os
import streamlit as st
from azure.storage.blob import BlockBlobService

# Azure Data Lake Gen1 connection details
account_name = '<your_account_name>'
account_key = '<your_account_key>'
container_name = '<your_container_name>'
blob_service = BlockBlobService(account_name=account_name, account_key=account_key)

# Local folder to download the images to
local_folder = './images'
os.makedirs(local_folder, exist_ok=True)

# Download the images from the Azure Data Lake Gen1 container to the local folder
blobs = blob_service.list_blobs(container_name)
image_blobs = [blob for blob in blobs if blob.name.endswith('.jpg') or blob.name.endswith('.jpeg') or blob.name.endswith('.png')]
for blob in image_blobs:
    blob_name = blob.name
    local_path = os.path.join(local_folder, blob_name)
    blob_service.get_blob_to_path(container_name, blob_name, local_path)

# Configure Streamlit to serve the images from the local folder
st.set_option('server.use_static_cache', False)

# Display the images on the Streamlit web application
st.title('Azure Data Lake Gen1 Images')
for blob in image_blobs:
    blob_name = blob.name
    local_path = os.path.join(local_folder, blob_name)
    st.image(local_path, caption=blob_name)













from PIL import Image

# Open the source and destination images
src_img = Image.open("source_image.jpg")
dest_img = Image.open("destination_image.jpg")

# Define the bounding box coordinates of the source image
src_bbox = (x, y, x + width, y + height)

# Crop the source image using the bounding box
cropped_img = src_img.crop(src_bbox)

# Define the bounding box coordinates of the destination image
dest_bbox = (dest_x, dest_y, dest_x + width, dest_y + height)
cropped_img = cropped_img.resize((dest_bbox[2]-dest_bbox[0], dest_bbox[3]-dest_bbox[1]))

# Get the average color of the destination bounding box
box_avg_color = dest_img.crop(dest_bbox).resize((1, 1)).getpixel((0, 0))

# Blend the cropped image with the destination image based on the average color of the destination bounding box
blended_img = Image.blend(cropped_img, dest_img.crop(dest_bbox), 0.5 * (box_avg_color[0] + box_avg_color[1] + box_avg_color[2]) / 765)


# Paste the cropped image onto the destination image
dest_img.paste(cropped_img, dest_bbox)

# Save the new image
dest_img.save("output_image.jpg")




















print("start")
import win32api, win32con
import time

n = 100

def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

for i in range(1000):
    print(i)
    time.sleep(60*1)
    click(10,10)
    time.sleep(60*1)
    click(80,80)
    time.sleep(60*1)
    click(1000,1000)










def cagr(start_value, end_value, num_periods):
    return (end_value / start_value) ** (1 / (num_periods - 1)) - 1


cagr_values = []
for i in df.columns:
    start_value = float(df[i].loc['2018'])
    end_value = float(df[i].loc['2016'])
    num_periods = len(df[i].loc['2016':'2018'])
    cagr_values.append(cagr(start_value, end_value, num_periods))

pd.concat([df, pd.DataFrame([cagr_values], columns=df.columns)], index='cagr')

















req_l = []
for v in value:
    start = v
    end = ""
    print(v)
    l1_index = l1.index(v)
    for j in l1[l1_index:]:
        if j in value:
            end = j
            if j != v:
                value.remove(j)
        else:
            if not end:
                end = start
            req_l.append(start+"-"+end)
            break
		
		
		
		
		
		
import os
import pandas as pd

# set the path to your file location
path = r'path\to\Text'
# create a empty list, where you store the content
list_of_text = []

# loop over the files in the folder
for file in os.listdir(path):
    # open the file
    with open(os.path.join(path, file)) as f:
        text = f.read()
    # append the text and filename
    list_of_text.append((text, file))

# create a dataframe and save
df = pd.DataFrame(list_of_text, columns = ['Text', 'Filename'])
df.to_csv(os.path.join(path, 'new_csv_file.csv'))












padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)




query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
  
# Initialize the ORB detector algorithm
orb = cv2.ORB_create()
  
# Now detect the keypoints and compute
# the descriptors for the query image
# and train image
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)




def feature_detection():
    st.subheader('Feature Detection in images')
    st.write("SIFT")
    image = load_image("tom1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()    
    keypoints = sift.detect(gray, None)
     
    st.write("Number of keypoints Detected: ",len(keypoints))
    image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(image, use_column_width=True,clamp = True)








# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = apple_img.reshape((-1,3))
print(pixel_vals.shape, type(pixel_vals[0][0]))

# Convert to float type
import numpy as np
pixel_vals = np.float32(pixel_vals)
print(pixel_vals.shape, type(pixel_vals[0][0]))

#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
# then perform k-means clustering wit h number of clusters defined as 3
#also random centres are initially chosen for k-means clustering
k = 9
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((apple_img.shape))






def controller(img, brightness=255,
               contrast=127):
   
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
 
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
 
    if brightness != 0:
 
        if brightness > 0:
 
            shadow = brightness
 
            max = 255
 
        else:
 
            shadow = 0
            max = 255 + brightness
 
        al_pha = (max - shadow) / 255
        ga_mma = shadow
 
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
 
    else:
        cal = img
 
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
 
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)
 
    # putText renders the specified text string in the image.
    cv2.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
    return cal





def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

frame = increase_brightness(frame, value=20)











import cv2
import numpy as np


def drawBox(boxes, image):
    for i in range(0, len(boxes)):
        # changed color and width to make it visible
        cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cvTest():
    # imageToPredict = cv2.imread("img.jpg", 3)
    imageToPredict = cv2.imread("49466033\\img.png ", 3)
    print(imageToPredict.shape)

    # Note: flipped comparing to your original code!
    # x_ = imageToPredict.shape[0]
    # y_ = imageToPredict.shape[1]
    y_ = imageToPredict.shape[0]
    x_ = imageToPredict.shape[1]

    targetSize = 416
    x_scale = targetSize / x_
    y_scale = targetSize / y_
    print(x_scale, y_scale)
    img = cv2.resize(imageToPredict, (targetSize, targetSize));
    print(img.shape)
    img = np.array(img);

    # original frame as named values
    (origLeft, origTop, origRight, origBottom) = (160, 35, 555, 470)

    x = int(np.round(origLeft * x_scale))
    y = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale))
    # Box.drawBox([[1, 0, x, y, xmax, ymax]], img)
    drawBox([[1, 0, x, y, xmax, ymax]], img)


cvTest()




















def rescaling_cords(img, x0_original, y0_original):
	scale_x = 2.5
	scale_y = 2
	resized_img = cv2.resize(img, [int(cols*scale_x), int(rows*scale_y)], interpolation=cv2.INTER_NEAREST)
	resized_rows, resized_cols = resized_img.shape[0:2] # cols = 800, rows = 512



	# Compute center column and center row
	x_original_center = (cols-1) / 2 # 159.5
	y_original_center = (rows-1) / 2 # 127.5



	# Compute center of resized image
	x_scaled_center = (resized_cols-1) / 2 # 399.5
	y_scaled_center = (resized_rows-1) / 2 # 255.5



	# Compute the destination coordinates after resize
	x1_scaled = (x0_original - x_original_center)*scale_x + x_scaled_center # 399.5
	y1_scaled = (y0_original - y_original_center)*scale_y + y_scaled_center # 255.5
	return int(x1_scaled), int(y1_scaled)



















img = cv.imread('sudoku.png')
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')









import streamlit as st
import streamlit.components.v1 as components
from tensorboard import manager
import shlex
import random
import html
import json
from pathlib import Path


def st_tensorboard(logdir="/logs/", port=6006, width=None, height=800, scrolling=True):
    """Embed Tensorboard within a Streamlit app
    Parameters
    ----------
    logdir: str
        Directory where TensorBoard will look to find TensorFlow event files that it can display.
        TensorBoard will recursively walk the directory structure rooted at logdir, looking for .*tfevents.* files.
        Defaults to /logs/
    port: int
        Port to serve TensorBoard on. Defaults to 6006
    width: int
        The width of the frame in CSS pixels. Defaults to the report’s default element width.
    height: int
        The height of the frame in CSS pixels. Defaults to 800.
    scrolling: bool
        If True, show a scrollbar when the content is larger than the iframe.
        Otherwise, do not show a scrollbar. Defaults to True.
    Example
    -------
    >>> st_tensorboard(logdir="/logs/", port=6006, width=1080)
    """

    logdir = Path(str(logdir)).as_posix()
    port = port
    width = width
    height = height

    frame_id = "tensorboard-frame-{:08x}".format(random.getrandbits(64))
    shell = """
        <iframe id="%HTML_ID%" width="100%" height="%HEIGHT%" frameborder="0">
        </iframe>
        <script>
        (function() {
            const frame = document.getElementById(%JSON_ID%);
            const url = new URL(%URL%, window.location);
            const port = %PORT%;
            if (port) {
            url.port = port;
            }
            frame.src = url;
        })();
        </script>
    """

    args_string = f"--logdir {logdir} --port {port}"
    parsed_args = shlex.split(args_string, comments=True, posix=True)
    start_result = manager.start(parsed_args)

    if isinstance(start_result, manager.StartReused):
        port = start_result.info.port
        print(f"Reusing TensorBoard on port {port}")

    proxy_url = "http://localhost:%PORT%"

    proxy_url = proxy_url.replace("%PORT%", "%d" % port)
    replacements = [
        ("%HTML_ID%", html.escape(frame_id, quote=True)),
        ("%JSON_ID%", json.dumps(frame_id)),
        ("%HEIGHT%", "%d" % height),
        ("%PORT%", "0"),
        ("%URL%", json.dumps(proxy_url)),
    ]

    for (k, v) in replacements:
        shell = shell.replace(k, v)

    return components.html(shell, width=width, height=height, scrolling=scrolling)






























































def IOU(box1, box2):
	""" We assume that the box follows the format:
		box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
		where (x1,y1) and (x3,y3) represent the top left coordinate,
		and (x2,y2) and (x4,y4) represent the bottom right coordinate """
	x1, y1, x2, y2 = box1	
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union
	return iou






















def run(bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou
























def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

import ast

iou = []
m1_match_m2_new= []
for count,m1 in enumerate(df[df['M1_converted'].notna()]['M1_converted']):
    m1_box = ast.literal_eval(m1)
    iou_lst = []
    for c,m2 in enumerate(df[df['M2_converted'].notna()]['M2_converted']):
        m2_box = ast.literal_eval(m2)
        correct = bb_intersection_over_union(m1_box, m2_box)
        iou_lst.append(correct)
#         print(correct)

    val = max(iou_lst)
    iou.append(val)
    index = iou_lst.index(val)
    m1_match_m2 = df['M2_converted'][index]
    
    if val < 0.8:
        val = 'no match'
        index = 'no match'
        m1_match_m2 = 'no match'
        
        
*************************************************************************************************************************************8
        
        
def distance2(boxA, boxB):
    return(list(np.abs(np.array(boxA)- np.array(boxB))))

m2_match_m3=[] 
for count, m2 in enumerate(df[df['M2_converted'].notna()]['M2_converted']):
    m2_box = ast.literal_eval(m2)
    iou_lst =[]
    diff = []
    min_list = []
    
    for c,m3 in enumerate(df[df['M3_converted'].notna()]['M3_converted']):
        m3_box = ast.literal_eval(m3)
        correct = distance2(m2_box, m3_box)    
        if all(i < 1000 for i in correct):
            min_list.append(m3_box)

    for j in min_list:
        correct_iou = bb_intersection_over_union(m2_box, j)
        iou_lst.append(correct_iou)

    val = max(iou_lst)
    index = iou_lst.index(val)
    match = min_list[index]
    m2_match_m3.append(match)
    
    print('shortlisted bboxes from stage1:',min_list)
    print('IOUs of above boxes:' , iou_lst)
    print('match:', match,'\n','*'*50)
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    print("val",val)
    print("index",index)
    
    m1_match_m2_new.append(m1_match_m2)
    print('m1_match_m2', m1_match_m2)
#     df['m1_match_m2'] = m1_match_m2
    print("++++++++++++++")


df['m1_match_m2'] = pd.Series(m1_match_m2_new)
df['iou'] = pd.Series(iou)

df


def distance(boxA, boxB):
    diff_sum = np.abs(sum(np.array(boxA) -  np.array(boxB)))
    return diff_sum



















intersection = numpy.logical_and(result1, result2)

union = numpy.logical_or(result1, result2)

iou_score = numpy.sum(intersection) / numpy.sum(union)

print(‘IoU is %s’ % iou_score)













import numpy as np
component1 = np.array([[0,1,1],[0,1,1],[0,1,1]], dtype=bool)
component2 = np.array([[1,1,0],[1,1,0],[1,1,0]], dtype=bool)

overlap = component1*component2 # Logical AND
union = component1 + component2 # Logical OR

IOU = overlap.sum()/float(union.sum()) 














def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

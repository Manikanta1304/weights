import os
os.environ["AZUREML_MODEL_DIR"] = '.'

# changing the current working directory to AICP_TOD folder in order to import necessary modules
parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(parent_dir_path))

from PIL import Image, ImageOps
import pandas as pd
import src.score_aml as score_aml
from streamlit_DOT.dot_api_call import api_call

import numpy as np
import json
import cv2
from datetime import date
from zipfile import ZipFile
import base64
import shutil
import json
import warnings
import time

# streamlit imports
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.shared import GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)       

# adding the application info
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
text-align: center;
opacity: 0.5;
}
</style>
<div class="footer">
<p>This app is only available between 04:30 - 21:30 France, 08:00 - 01:00 (next day) India, 22:30 - 15:30 (next day) US East Coast. Cisco <b>VPN</b> should be connected<a style='display: block; text-align: center;' target="_blank"></a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)


st.title("DOT code Detector")
st.write("Upload image of the sidewall of a tire to get the DOT code imprinted on its sidewall")
st.write("---")


def plot_dets(image , pred):
    all_poly_data =[]
    all_poly_data = pred["all_stage_output"]['l1_output']
    if all_poly_data: 
        ##plot only the first polygon
        poly = np.array(all_poly_data['0']['mask_poly'] , np.int32)
        image = cv2.polylines(image, [poly],
                      True, (0, 255, 0),
                      8)
                      
    for annotation in pred["all_stage_output"]["od_output"].values():
        image = cv2.rectangle( image,
                               (int(annotation["x1"]), int(annotation["y1"])),
                               (int(annotation["x2"]), int(annotation["y2"])), 
                               (0 , 0 , 255), 
                               3
                             )
        image = cv2.putText( image, 
                             annotation["classes_label"], 
                             (int(annotation["x1"]), int(annotation["y1"]) - 5), 
                             cv2.FONT_HERSHEY_SIMPLEX, 
                             2, 
                             (0 , 0 , 255), 
                             2, 
                             cv2.LINE_AA
                           )

    return image
    



st.cache(allow_output_mutation=True)
def get_predictions(selected_files):
    # initiating the score_aml which loads the models once
    if 'score_aml' not in st.session_state:
        st.session_state['score_aml'] = score_aml
        st.session_state['score_aml'].init()
    # st.write('outside***********', selected_files)
    with st.spinner(text="Running detections..."):
        predictions_df = pd.DataFrame(columns=['file_name', 'response']) 
        # proc_time = pd.DataFrame(columns=['Image_ID','Image_processing_time(secs)']) # processing time 
        # st.write('inside***********', selected_files)
        for image in selected_files:
            print(f"Image***\n{image['file_name']}")
            if image['file_name'] not in st.session_state['detected_imgs']:

                # Encode image in base64 and format request/params as json
                try:
                    file_index = st.session_state['files_dict'][image['file_name']]
                    b64_string = base64.b64encode(st.session_state['directory'][file_index].read()).decode('utf-8')
                    # print(b64_string)
                except:
                    #st.write('****************image not generated*****************')
                    with open('sample_image.jpg', "rb") as img_file:
                        b64_string = base64.b64encode(img_file.read()).decode('utf-8')

                st.session_state['detected_imgs'].append(image['file_name'])

            req_ref_key = "detect-cli"
            req_ref_val = os.path.splitext(os.path.basename(image['file_name']))[0]+ '_'+ str(int(time.time()))
            raw_data = json.dumps({"data":b64_string, "model":"scoreensemble", 
                                   "req_ref_key":req_ref_key, "req_ref_val":req_ref_val})

            # Run inference
            return_obj = st.session_state['score_aml'].run(raw_data)

            # Append the detected dot literals and the processing time to the dataframe
            predictions_df.loc[len(predictions_df)] = [image['file_name'], return_obj['dot']]

        # saving predictions to csv file. 
        # predictions_df.to_csv("predictions.csv", index=False)
        merged_df = pd.merge(st.session_state['input_df'], predictions_df, 
                                on=['file_name'], how='left', suffixes=('_',''))
        try: merged_df['response'] = merged_df.pop('response_').fillna(merged_df['response']) 
        except: pass
        st.session_state['merged_df'] = merged_df
        st.session_state['input_df'] = merged_df
        zip_href = save_detections()
        st.session_state['zip_href'] = zip_href

def run_api(selected_files):
    with st.spinner(text="Running api call..."):
        for image in selected_files:
            if image['file_name'] not in st.session_state['detected_imgs']:
                file_index = st.session_state['files_dict'][image['file_name']]
                if uploaded_file:
                    api_df = api_call([st.session_state['directory'][file_index]])
                else:
                    api_df = api_call(['sample_image.jpg'])
                merged_df = pd.merge(st.session_state['input_df'], api_df,  on=['file_name'], how='left', suffixes=('_',''))
                try: merged_df['response'] = merged_df.pop('response_').fillna(merged_df['response']) 
                except: pass
                st.session_state['merged_df'] = merged_df
                st.session_state['input_df'] = merged_df
                st.session_state['detected_imgs'].append(image['file_name'])


def createInputGrid():
    uploaded_files = st.session_state.get('directory', [])
    files_dict = {j.name:i for i,j in enumerate(uploaded_files)}
    input_df = pd.DataFrame({'file_name': files_dict.keys()})
    st.session_state['files_dict'] = files_dict
    if 'merged_df' in st.session_state:
        st.session_state['input_df'] = pd.merge(input_df, st.session_state['merged_df'], 
                                                on=['file_name'], how='left')
    else: st.session_state['input_df'] = input_df


def get_zip_file(ZipfileDotZip):
    with open(ZipfileDotZip,"rb") as f:
        bytes_ = f.read()
        b64 = base64.b64encode(bytes_).decode()
        href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}'>Click to download the detections</a>"
        shutil.rmtree(st.session_state['det_directory'])
    return href
    
    
def save_detections():     
    # creating a temporary directory
    dets_directory = 'detections'
    os.makedirs(dets_directory, exist_ok=True)
    st.session_state['det_directory'] = dets_directory
    
    # creating zip object to write the detections
    zipObj = ZipFile("detections.zip","w")

    # looping over the uploaded images to overlay the predictions and zip them
    for image in st.session_state['directory']:
        # st.session_state['detected_imgs'].append(img['file_name'])
        jsons = os.listdir(f'./40_project/TOT/model_output/req_output/{date.today()}')
        image_name = image.name.split('.')[0]
        file_index = -1
        # json_list = []
        for i in range(len(jsons)):
            if image_name=='_'.join(jsons[i].split('_')[1:-1]):
                file_index = i
                # json_list = jsons[i]
                break
            else: file_index = None
 
        # st.write(image_name, jsons[i].split('_')[1], jsons[file_index], file_index)
        pred_path = f'./40_project/TOT/model_output/req_output/{date.today()}/{jsons[file_index]}'
        with open(pred_path,'r') as f:
            pred = json.load(f)
            
        # detected_img = cv2.imread(os.path.join(f"./{st.session_state['image_dir']}", image))
        file_index = st.session_state['files_dict'][image.name]
        image = Image.open(st.session_state['directory'][file_index])
        detected_img = np.array(image)
        out_img = plot_dets(detected_img , pred)
        pred_text = pred['dot'] #str(pred_df[pred_df['Image_ID']==image_name]['Prediction'].values[0])
        out_img = cv2.putText(out_img, pred_text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
        cv2.imwrite(os.path.join(f'{dets_directory}', image_name +'.jpg'), out_img)
        
        # zip images
        zipObj.write(os.path.join(f'{dets_directory}', image_name +'.jpg'))
    # zipObj.write('predictions.csv')

    zipObj.close()   
    ZipfileDotZip = "detections.zip"
    zip_href = get_zip_file(ZipfileDotZip)
    os.remove('detections.zip')
    return zip_href    


# build aggrid out of dataframe response
def build_aggrid(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum")
    gb.configure_selection(use_checkbox=True, selection_mode='multiple')
    gb.configure_grid_options(domLayout='normal')
    gb.configure_column('file_name', headerCheckboxSelection = True)
    gridOptions = gb.build()
    data = AgGrid(df, 
                gridOptions=gridOptions, 
                enable_enterprise_modules=True, 
                allow_unsafe_jscode=True, 
                update_mode=GridUpdateMode.SELECTION_CHANGED, height=300, width='100%')
    return data


# main streamlit code
sidebar_col, grid_col, detect_col = st.columns([0.7, 2, 2])

def reload():
    if 'local_df' in st.session_state:
        del st.session_state['local_df']
    if 'api_df' in st.session_state:
        del st.session_state['api_df'] 
    if 'detected_imgs' in st.session_state:
        del st.session_state['detected_imgs']
    if 'merged_df' in st.session_state:
        del st.session_state['merged_df']   
    #if 'input_df' in st.session_state:
        #del st.session_state['input_df']

    createInputGrid()
    # st.experimental_rerun()


def sample_image_grid():
    #with grid_col:
        files_dict = {j:i for i,j in enumerate(['sample_image.jpg'])}
        sample_df = pd.DataFrame({'file_name': files_dict.keys()})
        st.session_state['files_dict'] = files_dict
        st.session_state['input_df'] = sample_df
        st.write('Select the sample file to detect and view the predictions')
        data = build_aggrid(sample_df)
        st.info("**To export the above table: right click on any cell --> Export --> CSV/Excel**")
        return data
        

def page_refresh():
    st.experimental_rerun()

with sidebar_col:
    method = st.radio(
        "Select option",
        ('Local', 'API call'), horizontal=True, on_change=reload, key='page_refresh')

# if selected local
if method == 'Local':
    with sidebar_col:        
        uploaded_file = st.file_uploader("Upload image", accept_multiple_files=True, 
                                                    type=['png', 'jpeg', 'jpg', 'JPG'], on_change=createInputGrid, 
                                                    key='directory')

    # displaying sample image on the grid
    data = None
    if not uploaded_file:
        with grid_col:
            sample_img = cv2.imread('sample_image.jpg')
            data = sample_image_grid()

    # data = None
    zip_href = None
    if uploaded_file and 'input_df' in st.session_state:
        with grid_col:
            st.write('Select the file name to detect and view the predictions')
            data = build_aggrid(st.session_state['input_df'])
            st.info("**To export the above table: right click on any cell --> Export --> CSV/Excel**")

            if st.button('Refresh grid'):
                st.session_state['input_df'] = st.session_state['input_df']


        if 'zip_href' in st.session_state:
            with sidebar_col:
                st.markdown(st.session_state['zip_href'], unsafe_allow_html=True)

    # selecting the image and sending for inference
    if 'detected_imgs' not in st.session_state:
        st.session_state['detected_imgs'] = []

    with detect_col:
        if data and data["selected_rows"]:
            selected_rows = data["selected_rows"]

            # if images are not in session state of detected images, run the predictions
            if any(img['file_name'] not in st.session_state['detected_imgs'] for img in selected_rows):
                if st.button('Detect'):
                    get_predictions(selected_rows) #[img['file_name']])
            
            # else, display the detections
            for img in selected_rows:
                if img['file_name'] in st.session_state['detected_imgs']:
                    # read the image and draw the bboxes
                    try:
                        file_index = st.session_state['files_dict'][img['file_name']]
                        image = ImageOps.exif_transpose(Image.open(st.session_state['directory'][file_index]))
                        detected_img = np.array(image)
                    except:
                        detected_img = sample_img.copy()

                    # st.session_state['detected_imgs'].append(img['file_name'])
                    jsons = os.listdir(f'./40_project/TOT/model_output/req_output/{date.today()}')
                    base_image_name = img['file_name'].split('.')[0]
                    json_index = -1
                    json_list = []
                    for i in range(len(jsons)):
                        if base_image_name=='_'.join(jsons[i].split('_')[1:-1]):
                            json_index = i
                            json_list = jsons[i]
                            break
                        else: json_index = None
                            
                            
                    pred_path = f'./40_project/TOT/model_output/req_output/{date.today()}/{jsons[json_index]}'
                    if os.path.exists(pred_path):
                        with open(pred_path,'r') as f:
                            pred = json.load(f)
                        out_img = plot_dets(detected_img , pred)
                        pred_text = pred['dot']
                        out_img = cv2.putText(out_img, pred_text,(100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
                        st.image(out_img, channels='BGR', caption=img['file_name'])
                        cv2.imwrite('./predictions/' + base_image_name +'.jpg', out_img)
                    else:
                        st.warning('No predictions generated for the selected image')


    

# api call option is selected
elif method == 'API call':
    data = None
    with sidebar_col:
        uploaded_file = st.file_uploader("Upload image", 
                                     accept_multiple_files=True, 
                                     type=['png', 'jpeg', 'jpg', 'JPG'], 
                                     on_change=createInputGrid, key='directory')

    # displaying sample image on the grid
    data = None
    if not uploaded_file:
        with grid_col:
            sample_img = cv2.imread('sample_image.jpg')
            data = sample_image_grid()


    if 'input_df' in st.session_state and uploaded_file: 
        with grid_col:
            st.write('Select the file name to detect and view the predictions')
            data = build_aggrid(st.session_state['input_df'])
            st.info("**To export the above table: right click on any cell --> Export --> CSV/Excel**")
            if st.button('Refresh grid'):
                st.session_state['input_df'] = st.session_state['input_df']

    if 'detected_imgs' not in st.session_state:
        st.session_state['detected_imgs'] = []


    with detect_col:
        if data and data["selected_rows"]:
            selected_rows = data["selected_rows"]
            if any(img['file_name'] not in st.session_state['detected_imgs'] for img in selected_rows):
                if st.button('Detect'):
                    run_api(selected_rows)

            for img in selected_rows:
                if img['file_name'] in st.session_state['detected_imgs']:
                    try: 
                        file_index = st.session_state['files_dict'][img['file_name']]
                        image = Image.open(st.session_state['directory'][file_index])
                        detected_img = np.array(image)
                    except:
                        detected_img = sample_img.copy()

                    result_df = st.session_state['merged_df']
                    try:
                        st.write(result_df)
                        code = result_df[result_df['file_name']==img['file_name']]['response'].values[0]['dot']
                        out_img = cv2.putText(detected_img, code,(100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                        st.image(out_img, caption=img['file_name'])
                    except:
                        st.image(detected_img, caption=img['file_name'])

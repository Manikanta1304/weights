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

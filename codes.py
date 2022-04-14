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

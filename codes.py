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


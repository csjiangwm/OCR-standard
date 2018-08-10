# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:42:42 2018

@author: jwm
"""
import cv2
import numpy as np
from matplotlib import cm
from math import fabs,sin,degrees,cos
import tensorflow as tf
import os
import threading

def normalize(data):
    if data.shape[0] == 0:
        return data
    max_ = data.max()
    min_ = data.min()
    return (data - min_) / (max_ - min_) if max_ - min_ != 0 else data - min_
   
def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)

 
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes   
    
def draw_boxes(im,bboxes,is_display=True,color=None,caption="Image",wait=True):
    """
        boxes: bounding boxes
        (x1,y1)                                (x2,y2)
            ——————————————————————————————————————
            \                                     \
            ——————————————————————————————————————
        (x3,y3)                                (x4,y4)
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)

    im = im.copy()
    index = 0
    for box in bboxes:
        if color == None:
            if len(box) == 8 or len(box) == 9:
                c = tuple(cm.jet([box[-1]])[0, 2::-1] * 255)
            else:
                c = tuple(np.random.randint(0, 256, 3))
        else:
            c = color

        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.line(im, (int(x1), int(y1)), (int(x3), int(y3)), c, 2)
        cv2.line(im, (int(x4), int(y4)), (int(x2), int(y2)), c, 2)
        cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4)), c, 2)
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index = index + 1
        # cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), c,2)
        # cv2.waitKey(0)
        # cv2.imshow('kk', im)
    return text_recs, im
    
def draw_boxes_(img, boxes, scale):
    box_id = 0
    img = img.copy()
    text_recs = np.zeros((len(boxes), 8), np.int)
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue

        if box[8] >= 0.8:
            color = (255, 0, 0)  # red
        else:
            color = (0, 255, 0)  # green

        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

        for i in range(8):
            text_recs[box_id, i] = box[i]

        box_id += 1

#    img = cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    return text_recs, img
    
    
def sort_box(box):
    """
    对box排序,及页面进行排版
    text_recs[index, 0] = x1
        text_recs[index, 1] = y1 !
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2 !
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3 !
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4 !
    """

    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box
    
def merge_box_old(box):
    '''
       merge and sort boxes that are in one line
    '''
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]])) # sort box according to page information
    y_sums = map(lambda x: sum([x[1], x[3], x[5], x[7]]),box)
    one_line = []
    for index,y_sum in enumerate(y_sums):
        try:
            if y_sums[index+1] - y_sum < 20:
                one_line.append(index)
                box[index+1][0] = min(box[index+1][0],box[index][0])
                box[index+1][2] = max(box[index+1][2],box[index][2])
                box[index+1][4] = min(box[index+1][4],box[index][4])
                box[index+1][6] = max(box[index+1][6],box[index][6])
        except IndexError:
            break
    for index in sorted(one_line,reverse=True):
        del(box[index])
    return box
    
def merge_box(box):
    '''
       merge and sort boxes that are in one line
    '''
    h_sum = lambda x: sum([x[1],x[3],x[5],x[7]])
    box = sorted(box, key=h_sum) # sort box according to page information
    # merge the boxes that are in one line
    y_sums = map(h_sum,box)
    nrof_box = len(y_sums)
    nrof_move = 0
    for index in range(nrof_box):
        try:
            if  y_sums[index+1] - y_sums[index] < 30:
                box_index = index - nrof_move
                box[box_index+1][0] = min(box[box_index+1][0],box[box_index][0])
                box[box_index+1][2] = max(box[box_index+1][2],box[box_index][2])
                box[box_index+1][4] = min(box[box_index+1][4],box[box_index][4])
                box[box_index+1][6] = max(box[box_index+1][6],box[box_index][6])
                del box[box_index]
                nrof_move += 1
        except IndexError:
            break
    return box
        
def DumpRotateImage(img, radian, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    # prepare to rotate the image
    height_rotated = int(width * fabs(sin(radian)) + height * fabs(cos(radian)))
    width_rotated = int(height * fabs(sin(radian)) + width * fabs(cos(radian)))
    # According to the radian of detected text rotate image
    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degrees(radian), 1)
    mat_rotation[0, 2] += (width_rotated - width) / 2
    mat_rotation[1, 2] += (height_rotated - height) / 2
    img_rotation = cv2.warpAffine(img, mat_rotation, (width_rotated, height_rotated), borderValue=(255, 255, 255))
    # Obtain the area after rotating
    [[pt1[0]], [pt1[1]]] = np.dot(mat_rotation,np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(mat_rotation,np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = img_rotation.shape[:2]
    
    img_out = img_rotation[max( 1, int(pt1[1]) ) : min( ydim - 1, int(pt3[1]) ), 
                           max( 1, int(pt1[0]) ) : min( xdim - 1, int(pt3[0]) )] # height, width
    return img_out
    
def error():
    return (IOError, EnvironmentError, OSError, OverflowError, ImportError, ValueError, IndexError, MemoryError,
            NameError, RuntimeError, TabError, NotImplementedError, AttributeError)
            
def get_sess():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.83
    sess = tf.InteractiveSession(config=config)
    return sess
    
def create_dirs(path):
    if isinstance(path,list):
        for path_ in path:
            if not os.path.exists(path_):
                os.mkdir(path_)
    else:
        if not os.path.exists(path):
            os.mkdir(path)
            
            
            
            
class thread(threading.Thread):
    def __init__(self,func,args=(),name=''):
        super(thread,self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        try:
            return self.result
        except Exception as e:
            print e
            return None
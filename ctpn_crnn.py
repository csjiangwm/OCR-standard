# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:33:25 2018

@author: jwm
"""
import tensorflow as tf
import torch
from configs.config import cfg
from models.CTPN import CTPN
from models.VGG import VGG
from models.CRNN import CRNN
import numpy as np
from data_loader.data_loader import DataLoader
from data_loader.text_generator import TextGenerator
from utils.utils import draw_boxes,merge_box,DumpRotateImage
from PIL import Image
from math import atan2
import cv2
import time
from configs import keys
cfg.KEYS = keys.alphabet


def rotate(img, angle_detector):
    angle = angle_detector.predict(img=np.copy(img))
    print('The angel of this character is:', angle)
    im = Image.fromarray(img)
    print('Rotate the array of this img!')
    if angle == 90:
        im = im.transpose(Image.ROTATE_270)
    elif angle == 180:
        im = im.transpose(Image.ROTATE_180)
    elif angle == 270:
        im = im.transpose(Image.ROTATE_90)
    img = np.array(im)
    return img
    
def detect(img, data, ctpn, sess):
    blobs, im_scales, img, _ = data.get_blobs(img, None)
    boxes,scores = ctpn.predict(blobs, im_scales, img, sess)
    boxes = ctpn.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    text_recs, detected_img = draw_boxes(img, boxes, caption='im_name', wait=True, is_display=True)
    return text_recs, detected_img, img

def recognize(im, text_recs, model, data, adjust=False):
    text_recs = merge_box(text_recs)
    index = 0
    results = {}
    xDim, yDim = im.shape[1], im.shape[0]

    for index, rec in enumerate(text_recs):
        '''
        rec[0]:  x1
        rec[1]:  y1
        rec[2]:  x2
        rec[3]:  y2
        rec[4]:  x3
        rec[5]:  y3
        rec[6]:  x4
        rec[7]:  y4
        
        (0,1)                                   (2,3)
            ——————————————————————————————————————
            \                                     \
            ——————————————————————————————————————
        (4,5)                                   (6,7)
        '''
        results[index] = [rec,]
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = [max(1, rec[0] - xlength), max(1, rec[1] - ylength)]
            pt2 = [rec[2], rec[3]]
            pt3 = [min(rec[6] + xlength, xDim - 2),min(yDim - 2, rec[7] + ylength)]
            pt4 = [rec[4], rec[5]]
        else:
            pt1 = [max(1, rec[0]), max(1, rec[1])]
            pt2 = [rec[2], rec[3]]
            pt3 = [min(rec[6], xDim - 2), min(yDim - 2, rec[7])]
            pt4 = [rec[4], rec[5]]
#        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
        radian = atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

        PartImg = DumpRotateImage(im, radian, pt1, pt2, pt3, pt4)
#        if PartImg.shape[0] < 1 or PartImg.shape[1] < 1 or PartImg.shape[0] > PartImg.shape[1]:  # 过滤异常图片
#           continue
        
        image = Image.fromarray(PartImg).convert('L')
        
        sim_pred = model.recognize(image,data)
        results[index].append(sim_pred) 

    return results

def main():
    ctpn = CTPN(cfg)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ctpn.load_ckpt(sess) # ctpn load
    
    if torch.cuda.is_available() and cfg.ALL_GPU:
        crnn = CRNN(cfg, 32, 1, len(cfg.KEYS) + 1, 256, 1).cuda()
    else:
        crnn = CRNN(cfg, 32, 1, len(cfg.KEYS) + 1, 256, 1).cpu()
    crnn.eval()
    crnn.load_state_dict(torch.load(cfg.CRNN_MODEL)) # crnn load
    
    if cfg.ADJUST_ANGLE:
        angle_detector = VGG(cfg) # vgg load
        angle_detector.load_weights()
    
    data = DataLoader(cfg)
    text = TextGenerator(cfg)
    
#        image_path = raw_input("please input your image path and name:") # get image path
    image_path = str('/home/jwm/Desktop/OCR-standard/images/xuanye.jpg')
    img = data.load_data(image_path)
    t = time.time()  
    if cfg.ADJUST_ANGLE:
        img = rotate(img, angle_detector) # rotate image if necessary
#    img = cv2.resize(img, (2000,3000),interpolation=cv2.INTER_CUBIC)
    text_recs, detected_img, img = detect(img, data, ctpn, sess) # detect text
    results = recognize(img, text_recs, crnn, text, adjust=True) # recognize text
    print("It takes time:{}s".format(time.time() - t))
    for key in results:
        print(results[key][1])      
            
             
    
if __name__ == '__main__':
    main()
    


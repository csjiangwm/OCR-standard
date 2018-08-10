# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:55:04 2018

@author: jwm
"""

import tensorflow as tf
from configs.config import cfg
from models.CTPN import CTPN
from models.VGG import VGG
import numpy as np
from data_loader.data_loader import DataLoader
from utils.utils import draw_boxes
import cv2
from PIL import Image
import time

def main():
    ctpn = CTPN(cfg)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ctpn.load_ckpt(sess)
    
    if cfg.ADJUST_ANGLE:
        angle_detector = VGG(cfg)
        angle_detector.load_weights()
    
    data = DataLoader(cfg)
    img = data.load_data('images/xuanye.jpg')
    
    t = time.time()
    
    if cfg.ADJUST_ANGLE:
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
#    img = cv2.resize(img, (2000,3000),interpolation=cv2.INTER_CUBIC)
    blobs, im_scales, resized_img, scale = data.get_blobs(img, None)
    boxes,scores = ctpn.predict(blobs, im_scales, resized_img, sess)
    boxes = ctpn.detect(boxes, scores[:, np.newaxis], resized_img.shape[:2])
    text_recs, im = draw_boxes(resized_img, boxes, caption='im_name', wait=True, is_display=True)
#    text_recs = sort_box(text_recs)
    print("It takes time:{}s".format(time.time() - t))
#    cv2.imshow('img',im)
#    cv2.waitKey(0)
    cv2.imwrite('images/result.jpg',im)
    
if __name__ == '__main__':
    main()



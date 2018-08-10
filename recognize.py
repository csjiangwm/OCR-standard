# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:33:25 2018

@author: jwm
"""
from __future__ import unicode_literals

import tensorflow as tf
import torch
from configs.config import cfg
from models.CTPN import CTPN
from models.VGG import VGG
from models.CRNN import CRNN
import numpy as np
from data_loader.data_loader import DataLoader
from data_loader.text_generator import TextGenerator
from utils.utils import draw_boxes,merge_box,DumpRotateImage,error,thread
from utils.web_content import find_item
import time
from PIL import Image
from math import atan2
import difflib
import re
import cv2

find_list = ['企业名称', '统一社会信用代码']

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
    
def key_compile(key_word, ocr_line):
    pattern = re.compile(key_word)
    m = pattern.findall(ocr_line)
    if m:
        ocr_res = ocr_line.split(m[0])[-1]
        return ocr_res
    return None
    
def similar_code(index,code_list,similar_char):
    res = [char for char in code_list]
    res[index] = similar_char
    return res
def fuzzy_ratio(pred,code):
    codes = []
    codes.append(list(pred))
    for i,c in enumerate(pred):
        tmp_lists = []
        if c == '0':
            for code_list in codes:
                tmp_lists.append(similar_code(i,code_list,'o'))
                tmp_lists.append(similar_code(i,code_list,'O'))
        if c == '9':
            for code_list in codes:
                tmp_lists.append(similar_code(i,code_list,'g'))
        if c == '1':
            for code_list in codes:
                tmp_lists.append(similar_code(i,code_list,'l'))
                tmp_lists.append(similar_code(i,code_list,'I'))
#                tmp_lists.append(similar_code(i,code_list,'['))
        if c == 's':
            for code_list in codes:
                tmp_lists.append(similar_code(i,code_list,'5'))
                tmp_lists.append(similar_code(i,code_list,'S'))
        if c == 'v':
            for code_list in codes:
                tmp_lists.append(similar_code(i,code_list,'V')) 
        if c == 'z':
            for code_list in codes:
                tmp_lists.append(similar_code(i,code_list,'Z'))
        if c == 'w':
            for code_list in codes:
                tmp_lists.append(similar_code(i,code_list,'W'))  
        codes.extend(tmp_lists)
    max_ratio = 0.0
    for string in codes:
        ratio = difflib.SequenceMatcher(None,''.join(string),code).ratio()
        if max_ratio < ratio:
            max_ratio = ratio
    return max_ratio
    
def match(ocr_results,items,keyword_name=r'[\S]*[名]?[\S]*称[\-]?',
                            keyword_code=r'代码',
                            keyword_pattern=r'[\S]*[类]?[\S]*型',
                            keyword_address=r'[\S]*[住]?[\S]*[所址][\-]?',
                            keyword_representative=r'[\S]*代表人[\-]?',
                            keyword_capital=r'[\S]*资本[\-]?',
                            keyword_date=r'[\S]*[成]?[\S]*[立]?[\S]*日期[\-]?',
                            keyword_deadline=r'[\S]*[营]?[\S]*[业]?[\S]*期限[\-]?'):
    name_matched = False
    code_matched = False
    res = 'out:'
    for key in ocr_results:
#        print ocr_results[key][0]
#        print ocr_results[key][1]
        location = [str(i) for i in ocr_results[key][0]]
        name = key_compile(keyword_name, ocr_results[key][1])
        if name:
            res += '名称:' + name + '[' +  ','.join(location) + ']'
#            print name
            seq = difflib.SequenceMatcher(None,name.encode('utf-8'),items[0])
            ratio = seq.ratio()
#            print ("ratio is :",ratio)
            if seq.ratio() > cfg.MATCH_RATIO:
                name_matched = True
        code = key_compile(keyword_code, ocr_results[key][1])
        if code:
            res += '统一社会信用代码：' + code + '[' +  ','.join(location) + ']'
#            print code
#            seq = difflib.SequenceMatcher(None,code,items[1])
#            ratio = seq.ratio()
            ratio = fuzzy_ratio(code,items[1])
#            print ("ratio is :",ratio)
            if ratio > cfg.MATCH_RATIO:
                code_matched = True
                
        pattern = key_compile(keyword_pattern, ocr_results[key][1])
        if pattern:
#            print pattern
            res += '名称：' + pattern + '[' +  ','.join(location) + ']'
        address = key_compile(keyword_address, ocr_results[key][1])    
        if address:
#            print address
            res += '地址：'+address + '[' +  ','.join(location) + ']'
        representative = key_compile(keyword_representative, ocr_results[key][1])    
        if representative:
#            print representative
            res += '法定代表人：' + representative + '[' +  ','.join(location) + ']'
        capital = key_compile(keyword_capital, ocr_results[key][1])    
        if capital:
#            print capital
            res += '注册资本：' + capital + '[' +  ','.join(location) + ']'
        date = key_compile(keyword_date, ocr_results[key][1])    
        if date:
#            print date
            res += '成立日期：' + date + '[' +  ','.join(location) + ']'
        deadline = key_compile(keyword_deadline, ocr_results[key][1])    
        if deadline:
#            print deadline
            res += '营业范围：' + deadline + '[' +  ','.join(location) + ']'
            break
    print('name result:',name_matched)
    print('code result:',code_matched)
    return (name_matched and code_matched), res

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
    
    if True:
#        image_path = raw_input("please input your image path and name:") # get image path
        image_path = str('/home/jwm/Desktop/OCR-standard/images/jingan.jpg')
        target = 'http://www.sgs.gov.cn/notice/notice/view?uuid=PS0hIGEWVK6yuuKgg.R0_lMe3iFxFCOq' # get target
        if 1:
            items = find_item(find_list,target)
            img = data.load_data(image_path)
            t = time.time()    
            if cfg.ADJUST_ANGLE:
                img = rotate(img, angle_detector) # rotate image if necessary
#            img = cv2.resize(img, (2000,3000),interpolation=cv2.INTER_CUBIC)
            text_recs, detected_img, img = detect(img, data, ctpn, sess) # detect text
            result = recognize(img, text_recs, crnn, text, adjust=True) # recognize text
            
            is_match, res = match(result,items)
            if is_match:
                print('success')
            else:
                print('failed')
                
            print("It takes time:{}s".format(time.time() - t))
    print res
#            break
#        except error() as e:
#            print e
#            print("cannot match")
#            break
#            order = raw_input("quit? yes(y) or no(n)? :")
#            if order == 'q':
#                break

def multi_thread():
    target = 'http://www.sgs.gov.cn/notice/notice/view?uuid=PS0hIGEWVK6yuuKgg.R0_lMe3iFxFCOq' # get target
    t1 = thread(func=find_item,args=(find_list,target))
    t1.start()
    t1.join()
    
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
    
    image_path = str('/home/jwm/Desktop/OCR-standard/images/license_recog/license_company_3_1.png')
    try:
        img = data.load_data(image_path)
        t = time.time()    
        if cfg.ADJUST_ANGLE:
            img = rotate(img, angle_detector) # rotate image if necessary
        text_recs, detected_img, img = detect(img, data, ctpn, sess) # detect text
        result = recognize(img, text_recs, crnn, text, adjust=True) # recognize text
        items = t1.get_result()
        if match(result,items):
            print('success')
        else:
            print('failed')
            
        print("It takes time:{}s".format(time.time() - t))
            
    except error() as e:
        print e
        print("cannot match")
        
def multi_process():
    from multiprocessing import Pool
    pool = Pool(2)
    target = 'http://www.sgs.gov.cn/notice/notice/view?uuid=PS0hIGEWVK6yuuKgg.R0_lMe3iFxFCOq' # get target
    items = pool.apply_async(find_item,args=(find_list,target))
    pool.close()
    pool.join()
#    print items.get()z
    
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
    
    image_path = str('/home/jwm/Desktop/OCR-standard/images/jingan.jpg')
    try:
        img = data.load_data(image_path)
        t = time.time()    
        if cfg.ADJUST_ANGLE:
            img = rotate(img, angle_detector) # rotate image if necessary
        text_recs, detected_img, img = detect(img, data, ctpn, sess) # detect text
        result = recognize(img, text_recs, crnn, text, adjust=True) # recognize text
        items = items.get()
        if match(result,items):
            print('success')
        else:
            print('failed')
            
        print("It takes time:{}s".format(time.time() - t))
            
    except error() as e:
        print e
        print("cannot match")
             
    
if __name__ == '__main__':
    main()
    


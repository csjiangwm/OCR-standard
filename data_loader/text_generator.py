# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:43:36 2018

@author: jwm
"""

import os
import numpy as np
from PIL import Image


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = np.random.permutation(total)
        self.cur = 0
        
    def __call__(self, batchsize):
        r_n=[]
        if(self.cur + batchsize > self.total):
            r_n_1 = self.range[self.cur : self.total]
            r_n.extend(r_n_1)
            
            np.random.shuffle(self.range)
            
            self.cur = (self.cur + batchsize) - self.total
            r_n_2 = self.range[0 : self.cur]
            r_n.extend(r_n_2)
            return np.array(r_n)
        else:
            r_n = self.range[self.cur : self.cur + batchsize]
            self.cur = self.cur + batchsize
            return r_n

class TextGenerator():
    def __init__(self,config):
        self.config = config
        self.load_strLabel()
        
    def load_strLabel(self):
        self.alphabet = self.config.KEYS + u'卍'
        self.nrof_classes = len(self.alphabet) - 1
        self.dict = {}
        for i, char in enumerate(self.config.KEYS):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
            
    @staticmethod
    def read_file(filename):
        res_dict = {}
        res_key = []
        nrof_files = 0
        for line in open(filename,'r'):
            line_list = line.strip().split(' ')
            res_dict[line_list[0]] = line_list[1:]
            res_key.append(line_list[0])
            nrof_files += 1
        return res_dict, np.array(res_key), nrof_files
        
    def encode(self, text, depth=0):
        """Support batch or single str."""
        length = []
        result = []
        for str in text:
            str = unicode(str, "utf8")
            length.append(len(str))
            for char in str:
                # print(char)
                index = self.dict[char]
                result.append(index)
        return (result, length)
        
    def decode_crnn(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
#                        print self.alphabet[t[i] - 1]
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode_crnn(t[index:index + l], l, raw=raw))
                index += l
            return texts
            
    def decode_densenet(self,pred):
        char_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != self.nrof_classes - 1:
                if not (i > 0 and pred_text[i] == pred_text[i - 1]) or (i > 1 and pred_text[i] == pred_text[i - 2]):
                    char_list.append(self.alphabet[pred_text[i] + 1])
        return u''.join(char_list)
        
        
    def next_batch(self, batch_size):
        yield self.shuffle_index
        
        
    def generator(self, filename):
        batch_size = self.config.TRAIN.BATCH_SIZE
        img_height = self.config.TRAIN.IMAGE_SIZE[0]
        img_width  = self.config.TRAIN.IMAGE_SIZE[1]
        
        img_labels, img_names, nrof_imgs = self.read_file(filename)
        
        next_batch = random_uniform_num(nrof_imgs)
        
        _inputs = np.zeros((batch_size, img_height, img_width, 1), dtype=np.float)
        labels = np.ones([batch_size, self.config.TRAIN.LABEL_LENGTH]) * 10000
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        
        while True:
            shuffle_img = img_names[next_batch(batch_size)] # img_names should be np.array
#            print shuffle_img
#            import pdb
#            pdb.set_trace()
            for index, img_name in enumerate(shuffle_img):
                img = Image.open(os.path.join(self.config.DATA_DIR, 'images', img_name)).convert('L')
                img = np.array(img, 'f') / 255.0 - 0.5 # (32, 280)
                _inputs[index] = np.expand_dims(img, axis=2) # (32, 280, 1)
                
                img_label = img_labels[img_name]
                nrof_labels = len(img_label)  # 10
                label_length[index] = nrof_labels
                
                if nrof_labels < 0:
                    print 'len < 0', img_name
                    
                input_length[index] = img_width // 8 # 35
                labels[index,:nrof_labels] = [int(k)-1 for k in img_label]
            inputs = {'input' : _inputs,  # (N, 32, 280, 1)
                      'label' : labels, # (N, 10)
                      'input_length': input_length, # (N, 1) --> 35
                      'label_length': label_length} # (N, 1) --> 10 
            outputs = {'ctc': np.zeros([batch_size])}
            yield (inputs, outputs)
        
        
        
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:28:44 2018

@author: jwm
"""

from keras.layers.core import Dense,Dropout,Activation,Permute, Lambda
from keras.layers.convolutional import Conv2D,ZeroPadding2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input,Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras import backend as K
from PIL import Image
import numpy as np
from keras import losses


class DenseNet:
    '''
       Densenet for OCR
    '''
    def __init__(self, config, nrof_classes):
        self.config = config
        self.nrof_classes = nrof_classes
        self.input = Input(shape=(32,None,1),name='input')
        self.setup()
        self.model = Model(inputs=self.input,outputs=self.preds)
        
        if self.config.DEBUG:
            self.model.summary()
            
        if self.config.IS_TRAIN:
            self.labels = Input(shape=[None], name='label', dtype='float32')
            self.label_length = Input(shape=[1], name='label_length', dtype='int64')
            self.input_length = Input(shape=[1], name='input_length', dtype='int64')
        
    def setup(self):
        _weight_decay = 1e-4
        _nrof_fliters = 64
        _dropout_rate = 0.2
        # conv 64 5*5 s=2   --> (16, 140, 64)
        x = Conv2D(_nrof_fliters, (5,5), strides=(2,2), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(_weight_decay))(self.input) 
        #  64 + 8 * 8 = 128 --> (16, 140, 128)
        x, _nrof_fliters = self.dense_block(x, 8, _nrof_fliters, 8, None)   
        # 128               --> ( 8, 70, 128)         
        x, _nrof_fliters = self.transition_block(x, 128, _dropout_rate, 2, _weight_decay)  
        # 128 + 8 * 8 = 192 --> ( 8, 70, 192)    
        x, _nrof_fliters = self.dense_block(x, 8, _nrof_fliters, 8, None)    
        # 192 -> 128        --> ( 4, 35, 128)             
        x, _nrof_fliters = self.transition_block(x, 128, _dropout_rate, 2, _weight_decay)
        # 128 + 8 * 8 = 192 --> ( 4, 35, 192)
        x, _nrof_fliters = self.dense_block(x, 8, _nrof_fliters, 8, None)                   
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        #                   --> ( 35, 4, 192)
        x = Permute((2, 1, 3), name='permute')(x)
        #                   --> ( 35, 768), TimeDistributed apply each layer into every time-sequence
        # 将Flatten应用到输入的每一个时间步上， --> 相当于将x的第2和3维执行Flatten操作 == tf.reshape(x,[x_shape[0],x_shape[1],-1])
        x = TimeDistributed(Flatten(), name='flatten')(x)
        # 35个序列，每个序列做一个sofmax分类，因为训练数据一张图片（一行）有多个汉字
        self.preds = Dense(self.nrof_classes, name='out', activation='softmax')(x)

    def conv_block(self, _input, fliters, kernal_size=(3,3), dropout=None, use_bias=True, weight_decay=1e-4):
        x = BatchNormalization(axis=-1,epsilon=1.1e-5)(_input)
        x = Activation('relu')(x)
        x = Conv2D(fliters, kernal_size, kernel_initializer='he_normal', padding='same', use_bias=use_bias,
                   kernel_regularizer=l2(weight_decay))(x)
        if dropout:
            x = Dropout(dropout)(x)
        return x
    
    def dense_block(self, _input, repeats, fliters, grow_rate, dropout=0.2):
        x = _input
        for i in range(repeats):
            cb = self.conv_block(x, grow_rate, dropout=dropout, weight_decay=0.)
            x = concatenate([x,cb], axis=-1)
            fliters += grow_rate
        return x, fliters
        
    def transition_block(self, _input, fliters, dropout=None, pooltype=1, weight_decay=1e-4):
        x = self.conv_block(_input, fliters, (1,1), dropout, False, weight_decay)
    
        if(pooltype == 2):
            x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        elif(pooltype == 1):
            x = ZeroPadding2D(padding = (0, 1))(x)
            x = AveragePooling2D((2, 2), strides=(2, 1))(x)
        elif(pooltype == 3):
            x = AveragePooling2D((2, 2), strides=(2, 1))(x)
        return x, fliters
        
    def load(self):
        self.model.load_weights(self.config.DENSENET_MODEL)
        
    def build_loss(self):
        '''
           densenet + CTC for training
        '''
        y_true = self.labels
        y_pred = self.preds
        input_length = self.input_length
        label_length = self.label_length
        
        def ctc_lambda(args):
            return K.ctc_batch_cost(args[0], args[1], args[2], args[3])
    
        self.loss = Lambda(ctc_lambda, output_shape=(1,), name='ctc')([y_true,y_pred,input_length,label_length])
        self.model_ctc = Model(inputs=[self.input, y_true, input_length, label_length], outputs=self.loss)
        # Some explaination can be found in the bottom
        self.model_ctc.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
        
    def predict(self,img,data):
        width, height = img.size[0], img.size[1]
        scale = height * 1.0 / 32
        width = int(width / scale)
        img = img.resize([width, 32], Image.ANTIALIAS)
       
        img = np.array(img).astype(np.float32) / 255.0 - 0.5
        
        X = img.reshape([1, 32, width, 1])
        
        y_pred = self.model.predict(X)
        y_pred = y_pred[:, :, :]
    
#        out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
#        out = u''.join([characters[x] for x in out[0]])
        out = data.decode_densenet(y_pred)
        return out
        
# Explaination :
# 只要定义了entire_model优化过程中就会带入entire_model中所有的层进行优化，无须在意complie中的loss是哪个
# 如果complie中的loss是self.loss时，保存模型时会将self.loss一起保存，造成的结果就是在测试的时候重新载入
# 会因为找不到CTC而提示报错
# 从compile的源码可以找到为什么loss是字典
# 从keras.losses源码可以找到为什么字典的value是个lambda函数

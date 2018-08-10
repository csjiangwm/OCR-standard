# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:36:26 2018

@author: jwm
"""

import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input, VGG16
#from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD

class VGG:
    '''
       detect the image angle
    '''
    def __init__(self,config):
        self.config = config
        self.setup()
        
    def setup(self):
#        vgg = VGG16(weights=None, input_shape=(224, 224, 3))
#        x = vgg.layers[-2].output
#        predictions_class = Dense(4, activation='softmax', name='predictions_class')(x)
#        prediction = [predictions_class]
        vgg = VGG16(weights=None, input_shape=(224, 224, 3),classes=4)
        prediction = [vgg.layers[-1].output]
        self.model = Model(inputs=vgg.input, outputs=prediction)
        sgd = SGD(lr=0.00001, momentum=0.9)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        
    def load_weights(self):
        self.model.load_weights(self.config.VGG_MODEL)
        
    def predict(self,img):
        ROTATE = [0, 90, 180, 270]
        img = Image.fromarray(img).convert('RGB')
        w, h = img.size
        # crop the image
        # left up corner (int(0.1 * w), int(0.1 * h))
        # right bottom corner (w - int(0.1 * w), h - int(0.1 * h))
        xmin, ymin, xmax, ymax = int(0.1 * w), int(0.1 * h), w - int(0.1 * w), h - int(0.1 * h)
        img = img.crop((xmin, ymin, xmax, ymax))  # flip the image edge
        img = img.resize((224, 224))
        img = np.array(img)
        img = preprocess_input(img.astype(np.float32))
        pred = self.model.predict(np.array([img]))
        index = np.argmax(pred, axis=1)[0]
        return ROTATE[index]
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:30:50 2018

@author: jwm
"""
import os
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

class DenseNetTrainer:
    def __init__(self, config, data, net):
        self.config = config
        self.data = data
        self.net = net
        
    def load_model(self):
        model_path = self.config.PRETRAINED_MODEL
        if model_path and os.path.exists(model_path):
            print 'loading model...'
            self.net.model.load_weights(model_path)
            print 'done!'
            
    def train(self):
        self.net.build_loss()
        self.load_model()
        
        train_loader = self.data.generator(os.path.join(self.config.DATA_DIR,'train.txt'))
        test_loader  = self.data.generator(os.path.join(self.config.DATA_DIR, 'test.txt'))
        
        model_name = 'weights-densenet-{epoch:02d}-{val_loss:.2f}.h5'
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.config.SAVED_PATH,model_name),
                                     monitor='val_loss',
                                     save_best_only=False,
                                     save_weights_only=True)
                                     
        lr_schedule = lambda epoch: 0.0005 * 0.4**epoch
        learning_rate = np.array([lr_schedule(i) for i in range(10)])
        # adjust the learning rate every epoch
        changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        tensorboard = TensorBoard(log_dir=self.config.DENSENET_LOGGER, write_graph=True)
        print('-----------Start training-----------')
        self.net.model_ctc.fit_generator(train_loader,
                                         steps_per_epoch = 3607567 // self.config.TRAIN.BATCH_SIZE,
                                         epochs = 10,
                                         initial_epoch = 0,
                                         validation_data = test_loader,
                                         validation_steps = 36440 // self.config.TRAIN.BATCH_SIZE,
                                         callbacks = [checkpoint, earlystop, changelr, tensorboard])
    
    def train_epoch(self):
        pass
    
    def train_step(self):
        pass
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""
import os
from configs.config import cfg
import tensorflow as tf
#from tensorflow.python.ops import gen_logging_ops
#from tensorflow.python.framework import ops as _ops

class BaseTrain(object):
    def __init__(self, sess, model, data, logger):
        self.model = model
        self.sess = sess
        self.data = data
        self.logger = logger
        self.init_global_step()
        
    def save(self, nrof_iters, write_meta_graph=True, step=None):
        '''
           Save function that saves the checkpoint in the path defined in the config file
        '''
        filename = (cfg.TRAIN.MODEL +'_iter_{:d}'.format(nrof_iters))
        
        print("Saving model...")
        if write_meta_graph:
            filepath = os.path.join(cfg.SAVED_PATH, filename) # Attention: use relative path
            self.model.saver.save(self.sess, filepath)
            print('Wrote snapshot to: {:s}'.format(filepath))
        else:
            checkpoint_path = os.path.join(cfg.SAVED_PATH, filename)
            self.model.saver.save(self.sess, checkpoint_path, global_step=step, write_meta_graph=write_meta_graph)
            metagraph_filename = os.path.join(cfg.SAVED_PATH, filename + '.meta')
            if not os.path.exists(metagraph_filename):
                print('Saving metagraph')
                self.model.saver.export_meta_graph(metagraph_filename)
        
        
    def init_global_step(self):
        '''
           Just initialize a tensorflow variable to use it as epoch counter
        '''
        self.global_step = tf.Variable(0, trainable=False)
    
    
    def train(self):
        '''
           Train the model
           Note: It may be overrided by its children
        '''
        for cur_epoch in range(self.global_step.eval(self.sess), self.config.MAX_ITERS + 1, 1):
            loss = self.train_epoch(cur_epoch)
            self.global_step = tf.assign(self.global_step, self.global_step + 1)
            if cur_epoch > self.config.MAX_ITERS or loss < 0.0001:
                break

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
        

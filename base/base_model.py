# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:15:17 2018

@author: jwm
"""
from __future__ import division

import tensorflow as tf

class BaseModel(object):
    def __init__(self, config):
        '''
           Create the base model
           All the variables are created
        '''
        self.config = config

    def load_ckpt(self, sess):
        '''
           Load latest checkpoint from the experiment path defined in the config file
        '''
#         latest_checkpoint = tf.train.latest_checkpoint(self.config.CTPN_MODEL)
        ckpt = tf.train.get_checkpoint_state(self.config.CTPN_MODEL)
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading model checkpoint {} ...\n".format(ckpt.model_checkpoint_path))
            if self.config.DEBUG:
                reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
                var_to_shape_map = reader.get_variable_to_shape_map()
                for key in var_to_shape_map:
                    print("Tensor_name is : ", key)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return ckpt.model_checkpoint_path
        return False
        
    
    def init_saver(self):
        '''
           Initialize the tensorflow saver that will be used in saving the checkpoints.
        '''
        if self.config.IS_TRAIN:
            self.saver = tf.train.Saver(max_to_keep=1, write_version=tf.train.SaverDef.V2)
        else:
            self.saver = tf.train.Saver()

    def build_model(self):
        raise NotImplementedError
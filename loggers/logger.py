# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

import tensorflow as tf
import os
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.framework import ops as _ops


class Logger:
    def __init__(self, sess,config):
        self.sess = sess
        self.config = config
        self.init_writer()
        
    def init_writer(self):
        self.writer = tf.summary.FileWriter(logdir=self.config.CTPN_LOGGER, graph=tf.get_default_graph(), flush_secs=5)
        
    def init_summary(self,**kwargs):
        for name,summary in kwargs.iteritems():
            tf.summary.scalar(name, summary)
        summary_op = tf.summary.merge_all()
        # A simple graph for write image summary
        log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
        log_image_name = tf.placeholder(tf.string)
        # import tensorflow.python.ops.gen_logging_ops as logging_ops
        log_image = gen_logging_ops._image_summary(log_image_name, tf.expand_dims(log_image_data, 0), max_images=1)
        _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)
        return summary_op, log_image, log_image_data, log_image_name

    # it can summarize scalars and images.
    def summarize(self, summary, step):
        self.writer.add_summary(summary=summary, global_step=step)
    

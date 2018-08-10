# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

import tensorflow as tf
from configs.config import cfg
import os
from lib.timer import Timer
from base.base_train import BaseTrain
from data_loader.data_generator import DataGenerator
from tqdm import tqdm


class CTPNTrainer(BaseTrain):
    
    def __init__(self, sess, model, data, logger):
        super(CTPNTrainer,self).__init__(sess,model,data, logger)
        self.imdb = data.load_imdb('voc_2007_trainval')
        self.roidb = data.get_training_roidb(self.imdb)
        self.pretrained_model = cfg.PRETRAINED_MODEL if cfg.PRETRAINED_MODEL else None

#        print('Computing bounding-box regression targets...')
#        if cfg.TRAIN.BBOX_REG:
#            self.bbox_means, self.bbox_stds = data.add_bbox_regression_targets(self.roidb)
#        print('done')
        self.timer = Timer()
        
    def get_train_op(self,loss):
        
        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
        else:
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)

        if cfg.TRAIN.WITH_CLIP:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=self.global_step)
        else:
            train_op = opt.minimize(loss, global_step=self.global_step)
        return train_op, lr
        
    def load_model(self,restore):
        restore_iter = 0
        
        if self.pretrained_model is not None and not restore:
            try:
                print(('Loading pretrained model weights from {:s}').format(self.pretrained_model))
                self.model.load_npz(self.pretrained_model, self.sess, True)
            except:
                raise 'Check your pretrained model {:s}'.format(self.pretrained_model)

        # resuming a trainer
        if restore:
            ckpt_path = self.model.load_ckpt(self.sess)
            stem = os.path.splitext(os.path.basename(ckpt_path))[0]
            restore_iter = int(stem.split('_')[-1])
            self.sess.run(self.global_step.assign(restore_iter))
        return restore_iter

    def train(self, max_iters, restore=False):
        """Network training loop."""
        data_layer = DataGenerator(self.roidb, self.imdb.nrof_classes, self.data)
        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = self.model.build_loss(ohem=cfg.TRAIN.OHEM)
        summary_op, log_image, log_image_data, log_image_name = self.logger.init_summary(rpn_reg_loss=rpn_loss_box,
                                                                                         rpn_cls_loss=rpn_cross_entropy,
                                                                                         model_loss=model_loss,
                                                                                         total_loss=total_loss)
        train_op, lr = self.get_train_op(total_loss)
        # intialize variables
        self.sess.run(tf.global_variables_initializer())
        restore_iter = self.load_model(restore)
        fetch_list = [total_loss, model_loss, rpn_cross_entropy, rpn_loss_box, summary_op, train_op]
        
        print(restore_iter, max_iters)
        for _iter in range(restore_iter, max_iters, cfg.TRAIN.EPOCH_SIZE):
            losses = self.train_epoch(_iter,lr,data_layer,fetch_list)

            print('iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %f' % \
                  (_iter+cfg.TRAIN.EPOCH_SIZE, max_iters, losses[0], losses[1], losses[2], losses[3], losses[5].eval()))
            self.logger.summarize(losses[4],self.global_step.eval())    
            self.save(_iter+cfg.TRAIN.EPOCH_SIZE)
                
    def train_epoch(self, tm_iter, lr, data_layer, fetch_list):
        loop = tqdm(range(cfg.TRAIN.EPOCH_SIZE))
        for _iter in loop:
            tm_iter += _iter
            if tm_iter != 0 and tm_iter % cfg.TRAIN.STEPSIZE == 0:
                self.sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))
            self.timer.tic()
            total_loss_val, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val, summary_str, _ = \
                                                                                self.train_step(data_layer,fetch_list)
            _diff_time = self.timer.toc(average=False)
            
        print('speed: {:.3f}s / iter'.format(_diff_time))
        return total_loss_val, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val, summary_str, lr

    def train_step(self, data_layer, fetch_list):
        blobs = data_layer.forward()
        feed_dict = {self.model.data: blobs['data'],
                     self.model.im_info: blobs['im_info'],
                     self.model.keep_prob: 0.5,
                     self.model.gt_boxes: blobs['gt_boxes'],
                     self.model.gt_ishard: blobs['gt_ishard'],
                     self.model.dontcare_areas: blobs['dontcare_areas']}
        
        return self.sess.run(fetches=fetch_list, feed_dict=feed_dict)
        


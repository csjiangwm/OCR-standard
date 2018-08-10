# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from base.network import Network
from base.base_model import BaseModel
import tensorflow as tf
import numpy as np
from utils.anchors import nms
from utils.utils import normalize
from utils.proposal import TextProposalConnector

class CTPN(Network,BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self,config)
#        self.config = config
        self.inputs = []
        if self.config.IS_TRAIN:
            self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
            self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
            self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
            self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
            self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
            self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes, \
                                'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        else:
            self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
            self.layers = dict({'data': self.data, 'im_info': self.im_info})
            self.connector = TextProposalConnector()
        self.keep_prob = tf.placeholder(tf.float32)
        self.setup()
        self.init_saver()
        
    def setup(self):
        # anchor_scales = [8, 16, 32]
        anchor_scales = self.config.ANCHOR_SCALES
        _feat_stride = [16, ]
        # net frame
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1')
         .conv(3, 3, 64, 1, 1, name='conv1_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1')
         .conv(3, 3, 128, 1, 1, name='conv2_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3'))
        # ========= RPN ============
        (self.feed('conv5_3').conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')) # N x 14 x 14 x 512
        
        (self.feed('rpn_conv/3x3').Bilstm(512, 128, 512, name='lstm_o'))  # N x 14 x 14 x 512
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 4, name='rpn_bbox_pred')) # N x 14 x 14 x (1*10*4)
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 2, name='rpn_cls_score')) # N x 14 x 14 x (1*10*2)
        
#        if self.config.IS_TRAIN:
#            
#            # 给每个anchor上标签，并计算真值（也是delta的形式），以及内部权重和外部权重
#            (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
#             .anchor_target_layer(_feat_stride, anchor_scales, name='rpn-data'))
    
        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))
         
        if not self.config.IS_TRAIN:
            # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
            # A x 2 是每个box的background(bg)和forwardground（fg）得分
            (self.feed('rpn_cls_prob')
            .spatial_reshape_layer(len(anchor_scales) * 10 * 2, name='rpn_cls_prob_reshape'))
            
            (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
            .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))
        else:
            # generating training labels on the fly
            # output: rpn_labels(HxWxA, 2) rpn_bbox_targets(HxWxA, 4) rpn_bbox_inside_weights rpn_bbox_outside_weights
            (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, name='rpn-data'))
            
    def build_loss(self, ohem=False):
        # classification loss
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)
        fg_keep = tf.equal(rpn_label, 1)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1)) # ignore_label(-1)
        rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)  # shape (N, 2)
        rpn_label = tf.gather(rpn_label, rpn_keep)
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)
        # box loss
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape (1, H, W, Ax4)
        rpn_bbox_targets = self.get_output('rpn-data')[1]
        rpn_bbox_inside_weights = self.get_output('rpn-data')[2]
        rpn_bbox_outside_weights = self.get_output('rpn-data')[3]
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)  # shape (N, 4)
        rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
        rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
        rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)
        deltas = rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)
        rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * self.smooth_l1_dist(deltas), reduction_indices=[1])
        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
        
        model_loss = rpn_cross_entropy + rpn_loss_box

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss

        return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name):
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                    (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)
                    
    def predict(self, blobs, im_scales, image, sess):
        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],dtype=np.float32)
        feed_dict = {self.data: blobs['data'],self.im_info: blobs['im_info'], self.keep_prob: 1.0}  
            
        rois = sess.run([self.get_output('rois')[0]], feed_dict=feed_dict) # conf + box, deltas
        rois = rois[0] # conf + box
        scores = rois[:, 0] # scores
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]
        return boxes,scores
                    
    def detect(self, text_proposals, scores, size):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        """
        # 删除得分较低的proposal
        # text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds = np.where(scores > self.config.TEXT_PROPOSALS_MIN_SCORE)[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds] # choose that score > 0.7
        # 按得分排序
        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices] # sort
        # 对proposal做nms
        # nms for text proposals
        keep_inds = nms(np.hstack((text_proposals, scores)), self.config.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds] # nms
        # 获取检测结果
        scores = normalize(scores)
        text_lines = self.connector.get_text_lines(text_proposals, scores, size)
        # 过滤boxes
        keep_inds = self.filter_boxes(text_lines)
        text_lines = text_lines[keep_inds]
        # 对lines做nms
        if text_lines.shape[0] != 0:
            keep_inds = nms(text_lines, self.config.TEXT_LINE_NMS_THRESH)
            text_lines = text_lines[keep_inds]
        return text_lines
        
    def filter_boxes(self, boxes):
        heights = boxes[:, 3] - boxes[:, 1] + 1
        widths = boxes[:, 2] - boxes[:, 0] + 1
        scores = boxes[:, -1]
        return np.where((widths / heights > self.config.MIN_RATIO) & (scores > self.config.LINE_MIN_SCORE) &
                        (widths > (self.config.TEXT_PROPOSALS_WIDTH * self.config.MIN_NUM_PROPOSALS)))[0]
        
#    def filter_boxes(self, boxes):
#        heights=np.zeros((len(boxes), 1), np.float)
#        widths=np.zeros((len(boxes), 1), np.float)
#        scores=np.zeros((len(boxes), 1), np.float)
#        index=0
#        for box in boxes:
#            heights[index] = (abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1
#            widths[index] = (abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1
#            scores[index] = box[8]
#            index += 1
#
#        return np.where((widths / heights > 0.01) & (scores > 0.6) & (widths > (0)))[0]
                        
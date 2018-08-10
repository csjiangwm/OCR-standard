# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

import numpy as np
import cv2
from pascal_voc import pascal_voc
from lib.bbox import bbox_overlaps
from utils.anchors import bbox_transform
import os

class DataLoader(object):
    '''
       Generate data
    '''
    def __init__(self, config):
        self.config = config
        
    def load_data(self, path):
        if isinstance(path,str):
            data = cv2.imread(path)
#            print data.shape
        else:
            data = []
            for data_path in path:
                data.append(cv2.imread(data_path))
        return data
    
    def resize_im(self, im, scale, max_scale=None):
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, (0, 0), fx=f, fy=f), f
        # return cv2.resize(im, (0, 0), fx=1.2, fy=1.2), f
    
    
    def im_list_to_blob(self, ims):
        """Convert a list of images into a network input.
    
        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        blob = np.zeros((len(ims), max_shape[0], max_shape[1], 3),dtype=np.float32)
        for index,im in enumerate(ims):
            blob[index, 0:im.shape[0], 0:im.shape[1], :] = im
        return blob
        
    def _get_image_blobs(self, roidb, scale_inds):
        """Builds an input blob from the images in the roidb at the specified scales."""
        num_images = len(roidb)
        processed_ims = []
        im_scales = []
        for i in range(num_images):
            im = cv2.imread(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            im, im_scale = self._get_image_blob(im, scale_inds[i], False)
            im_scales.append(im_scale)
            processed_ims.append(im)
    
        # Create a blob to hold the input images
        blob = self.im_list_to_blob(processed_ims)
    
        return blob, im_scales
        
    def _get_image_blob(self, im, index = 0, copy = True):
        im = im.astype(np.float32, copy=copy)
        im -= self.config.PIXEL_MEANS
    
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
    
        if not self.config.IS_TRAIN:
            processed_ims = []
            im_scale_factors = []
            for target_size in self.config.TEST.SCALES:
                im_scale = float(target_size) / float(im_size_min)
                # Prevent the biggest axis from being more than MAX_SIZE
                if np.round(im_scale * im_size_max) > self.config.TEST.MAX_SIZE:
                    im_scale = float(self.config.TEST.MAX_SIZE) / float(im_size_max)
                if im_scale != 1.0:
                    im = cv2.resize(im,None,None,fx=im_scale,fy=im_scale,interpolation=cv2.INTER_LINEAR)
                im_scale_factors.append(im_scale)
                processed_ims.append(im)
            # Create a blob to hold the input images
            blob = self.im_list_to_blob(processed_ims)
            return blob, np.array(im_scale_factors)
        else:
            im_scale = float(self.config.TRAIN.SCALES[index]) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.config.TRAIN.MAX_SIZE:
                im_scale = float(self.config.TRAIN.MAX_SIZE) / float(im_size_max)
            if self.config.TRAIN.RANDOM_DOWNSAMPLE:
                r = 0.6 + np.random.rand() * 0.4
                im_scale *= r
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)
            return im, im_scale
            
    
    def get_blobs(self, img, rois):
        img, scale = self.resize_im(img, scale=self.config.SCALE, max_scale=self.config.MAX_SCALE)
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scale_factors = self._get_image_blob(img)
        return blobs, im_scale_factors,img, scale
        
    def load_imdb(self, name):
        names = name.split('_')
        year,image_set = names[1:]
        imdb = pascal_voc(image_set,year)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        return imdb
        
    def load_next_batch(self, roidb, num_classes):
        """Given a roidb, construct a minibatch sampled from it."""
        num_images = len(roidb)
        # Sample random scales to use for each image in this batch
        random_scale_inds = np.random.randint( 0, high=len(self.config.TRAIN.SCALES), size=num_images)
        assert (self.config.TRAIN.BATCH_SIZE % num_images == 0), 'num_images ({}) must divide BATCH_SIZE ({})'. \
                format(num_images, self.config.TRAIN.BATCH_SIZE)
    
        # Get the input image blob, formatted for caffe
        im_blob, im_scales = self._get_image_blobs(roidb, random_scale_inds)
    
        blobs = {'data': im_blob}
    
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['gt_ishard'] = roidb[0]['gt_ishard'][gt_inds] if 'gt_ishard' in roidb[0] \
                                                            else np.zeros(gt_inds.size, dtype=int)
        # blobs['gt_ishard'] = roidb[0]['gt_ishard'][gt_inds]
        blobs['dontcare_areas'] = roidb[0]['dontcare_areas'] * im_scales[0] if 'dontcare_areas' in roidb[0] \
                                                                            else np.zeros([0, 4], dtype=float)
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        blobs['im_name'] = os.path.basename(roidb[0]['image'])
    
        return blobs
        
    def get_training_roidb(self,imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        if self.config.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')
    
        print('Preparing training data...')
        if self.config.TRAIN.HAS_RPN:
            self.prepare_roidb(imdb)
        else:
            self.prepare_roidb(imdb)
        print('done')
    
        return imdb.roidb
        
    def prepare_roidb(self, imdb):
        """Enrich the imdb's roidb by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        sizes = imdb.get_sizes()
        roidb = imdb.roidb
        for i in range(imdb.nrof_images):
            roidb[i]['image'] = imdb.image_path_at(i)
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
            gt_overlaps = roidb[i]['gt_overlaps'].toarray() # need gt_overlaps as a dense array for argmax
            
            max_classes = gt_overlaps.argmax(axis=1) # gt class that had the max overlap
            roidb[i]['max_classes'] = max_classes
            
            max_overlaps = gt_overlaps.max(axis=1) # max overlap with gt over classes (columns)
            roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)
            roidb[i]['bbox_targets'] = self._compute_targets(roidb[i]['boxes'], max_overlaps, max_classes)
            
            
    def _compute_targets(self, rois, overlaps, labels):
        """
        Compute bounding-box regression targets for an image.
        for each roi find the corresponding gt_box, then compute the distance.
        """
        # Indices of ground-truth ROIs
        gt_inds = np.where(overlaps == 1)[0]
        if len(gt_inds) == 0:
            # Bail if the image has no ground-truth ROIs
            return np.zeros((rois.shape[0], 5), dtype=np.float32)
        # Indices of examples for which we try to make predictions
        ex_inds = np.where(overlaps >= self.config.TRAIN.BBOX_THRESH)[0]
    
        # Get IoU overlap between each ex ROI and gt ROI
        ex_gt_overlaps = bbox_overlaps(np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
                                       np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))
    
        # Find which gt ROI each ex ROI has max overlap with:
        # this will be the ex ROI's gt target
        gt_assignment = ex_gt_overlaps.argmax(axis=1)
        gt_rois = rois[gt_inds[gt_assignment], :]
        ex_rois = rois[ex_inds, :]
    
        targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
        targets[ex_inds, 0] = labels[ex_inds]
        targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
        return targets
        
#    def add_bbox_regression_targets(self,roidb):
#        """
#        Add information needed to train bounding-box regressors.
#        For each roi find the corresponding gt box, and compute the distance.
#        then normalize the distance into Gaussian by minus mean and divided by std
#        """
#        assert len(roidb) > 0
#        assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'
#    
#        num_images = len(roidb)
#        # Infer number of classes from the number of columns in gt_overlaps
#        num_classes = roidb[0]['gt_overlaps'].shape[1]
#    
#        if self.config.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
#            # Use fixed / precomputed "means" and "stds" instead of empirical values
#            means = np.tile(
#                np.array(self.config.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes, 1))
#            stds = np.tile(
#                np.array(self.config.TRAIN.BBOX_NORMALIZE_STDS), (num_classes, 1))
#        else:
#            # Compute values needed for means and stds
#            # var(x) = E(x^2) - E(x)^2
#            class_counts = np.zeros((num_classes, 1)) + self.config.EPS
#            sums = np.zeros((num_classes, 4))
#            squared_sums = np.zeros((num_classes, 4))
#            for im_i in range(num_images):
#                targets = roidb[im_i]['bbox_targets']
#                for cls in range(1, num_classes):
#                    cls_inds = np.where(targets[:, 0] == cls)[0]
#                    if cls_inds.size > 0:
#                        class_counts[cls] += cls_inds.size
#                        sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
#                        squared_sums[cls, :] += \
#                            (targets[cls_inds, 1:] ** 2).sum(axis=0)
#    
#            means = sums / class_counts
#            stds = np.sqrt(squared_sums / class_counts - means**2)
#            # too small number will cause nan error
#            assert np.min(stds) < 0.01, \
#                'Boxes std is too small, std:{}'.format(stds)
#        if self.config.DEBUG:
#            print('bbox target means:')
#            print(means)
#            print(means[1:, :].mean(axis=0))  # ignore bg class
#            print('bbox target stdevs:')
#            print(stds)
#            print(stds[1:, :].mean(axis=0))  # ignore bg class
#    
#        # Normalize targets
#        if self.config.TRAIN.BBOX_NORMALIZE_TARGETS:
#            print("Normalizing targets")
#            for im_i in range(num_images):
#                targets = roidb[im_i]['bbox_targets']
#                for cls in range(1, num_classes):
#                    cls_inds = np.where(targets[:, 0] == cls)[0]
#                    roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
#                    roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
#        else:
#            print("NOT normalizing targets")
#    
#        # These values will be needed for making predictions
#        # (the predicts will need to be unnormalized and uncentered)
#        return means.ravel(), stds.ravel()
#        
#    
#            
#        

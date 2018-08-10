# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:35:17 2018

@author: jwm
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from configs.config import cfg
from lib.bbox import bbox_overlaps, bbox_intersections
from lib.cython_nms import nms as cython_nms
try:
    from lib.gpu_nms import gpu_nms
except:
    gpu_nms = cython_nms
    
def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor

def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    for index,(h, w) in enumerate(sizes):
        anchors[index] = scale_anchor(base_anchor, h, w)
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],scales=2 ** np.arange(3, 6)):
    '''
       generate anchors that width is fixed to 16
    '''
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes, base_size)
    
def bbox_transform_inv(boxes, deltas):
    '''parameters
       boxes: anchors
       deltas: offsets
       return: (HxWxA,4)
    '''
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = ctr_x[:, np.newaxis]  # (HxWxA,1)
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
    
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
    
def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
    
def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS:
        try:
            return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
        except:
            return cython_nms(dets, thresh)
    else:
        return cython_nms(dets, thresh)
        
def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes , anchors
    :param gt_rois: n * 4 numpy array, ground-truth boxes, gt_boxes
    :return: targets: n * 4 numpy array, predicted ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
        'Invalid boxes found: {} {}'. format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets
    
def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        assert data.shape[0] == len(inds), 'number of data should be equal to the length of inds'
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image.
        parameters
        ----------
        ex_rois: anchors
        gt_rois: gt_boxes
    """

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
    
def shift_anchors(_anchors,height,width,_feat_stride,A):
    '''each feature in feature map generate A anchors
       Parameters
       ----------
       _anchors: (A,4) A anchors belong to one feature
       height: height of feature map
       width: width of feature map
       _feat_stride:
       A: number of anchors
       --------
       Returns:
       --------
       all_anchors: (A*Width*Height,4) all anchors belong to the feature map
                    Note: the position of anchors are mapped to original image
       nrof_anchors: number of all_anchors
    '''
    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride # [W,]
    shift_y = np.arange(0, height) * _feat_stride # [H,]
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # [W,H]
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel())).transpose()#[W*H,4]
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    K = shifts.shape[0]  # 50*37，feature-map's width x height
    all_anchors = (_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))) # (HxW) * A * 4
    nrof_anchors = int(K * A)
    all_anchors = all_anchors.reshape((nrof_anchors, 4)) # (HxWxA) x 4
    return all_anchors,nrof_anchors

    
def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride=[16, ], anchor_scales=[16, ]):
    """
    Parameters
    ----------
    rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, probality of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    cfg_key: 'TRAIN' or 'TEST'
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois/box : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
    box_deltas: (1 x Hx W x A, 4)

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)

    """
    cfg_key = cfg_key.decode('ascii')
    _anchors = generate_anchors(scales=np.array(anchor_scales))  # generate 9 anchors
    # print('anchors', _anchors)
    _num_anchors = _anchors.shape[0]  # number of anchors: 9

    im_info = im_info[0]  # [3,], the height, weight and scales of original image

    assert rpn_cls_prob_reshape.shape[0] == 1, 'Only single item batches are supported'

    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N # Number of top scoring boxes to keep before applying NMS to RPN proposals
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N # Number of top scoring boxes to keep after applying NMS to RPN proposals
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH  # nms threshold 0.7
    min_size = cfg[cfg_key].RPN_MIN_SIZE  # The minimize size of proposal box
    
    height, width = rpn_cls_prob_reshape.shape[1:3]  # height and width of feature-map

    # the first set of _num_anchors channels are bg probabilities, which we don't care
    # the second set are the fg probabilities, which we want (1, H, W, A), i.e., score of object,
    rpn_cls_prob_reshape = np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])
    scores = np.reshape(rpn_cls_prob_reshape[:, :, :, :, 1],[1, height, width, _num_anchors])

    bbox_deltas = rpn_bbox_pred  # relative location, It should be futher transformed to real location of original image
    # im_info = bottom[2].data[0, :]

    if cfg.DEBUG:
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))
        print('score map size: {}'.format(scores.shape))
        
    # 1. Generate proposals from bbox deltas and shifted anchors
    # generate the shift of anchor, which can be used to futher compute all the anchors in the whole image 
    anchors,_ = shift_anchors(_anchors,height,width,_feat_stride,_num_anchors) # return (HxWxA, 4)

    # Transpose and reshape predicted bbox transformations to get them into the same order as the anchors:
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.reshape((-1, 4))  # (HxWxA, 4)

    # Same story for the scores:
    scores = scores.reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    # achieve the real location of box in original image
    proposals = bbox_transform_inv(anchors, bbox_deltas)  

    # 2. clip predicted boxes that exceed the image, remain all the anchors!!!
    proposals = clip_boxes(proposals, im_info[:2])

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size * im_info[2])
    proposals = proposals[keep, :]  # save the left proposal
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep, :]

    # # remove irregular boxes, too fat too tall
    # keep = _filter_irregular_boxes(proposals)
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:  # keep 12000 proposal boxes for nms
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]
    bbox_deltas = bbox_deltas[order, :]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)),nms_thresh)  # nms，keeep 2000 proposal boxes
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep, :]

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    blob = np.hstack((scores.astype(np.float32, copy=False),proposals.astype(np.float32, copy=False)))

    return blob, bbox_deltas

def _compute_labels(A,anchors,gt_boxes, dontcare_areas, gt_ishard):
    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((A,), dtype=np.float32)
    labels.fill(-1) # initlize label to be -1
    # compute overlaps between the anchors and the gt boxes for labeling anchor overlaps, shape is A x G. 
    # Note: anchors (A,4), gt_boxes (G,5)
    overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                             np.ascontiguousarray(gt_boxes, dtype=np.float))  
    # (A)#找到和所有anchor的overlap最大的gt_box的index
    argmax_overlaps = overlaps.argmax(axis=1)  # the max index of each raw --> the index of gt_box
    # (A)#找到和所有anchor的overlap最大的gt_box的value
    max_overlaps = overlaps[np.arange(A), argmax_overlaps]
    # (G)#找到和所有gt_box的overlap最大的anchor的index
    gt_argmax_overlaps = overlaps.argmax(axis=0) # the max index of each column --> the index of anchor
    # (G)#找到和所有gt_box的overlap最大的anchor的value
    gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
    # (G)#找到和所有gt_box的overlap最大的anchor的index, 同时找到所有具有这些最大overlap的anchor
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    # assign bg labels first so that positive labels can clobber them
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 # max_overlaps and labels have same shape 
    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1
    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1  # overlap>0.7, fg

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        # 将所有的anchor中与gt_box的overlap最大值还小于0.3的anchor的label置为0
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 

    # preclude dontcare areas
    if dontcare_areas is not None and dontcare_areas.shape[0] > 0:
        # intersec shape is D x A
        intersecs = bbox_intersections(np.ascontiguousarray(dontcare_areas, dtype=np.float),
                                       np.ascontiguousarray(anchors, dtype=np.float))
        intersecs_ = intersecs.sum(axis=0)  # A x 1
        labels[intersecs_ > cfg.TRAIN.DONTCARE_AREA_INTERSECTION_HI] = -1

    # preclude hard samples that are highly occlusioned, truncated or difficult to see
    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
        assert gt_ishard.shape[0] == gt_boxes.shape[0]
        gt_ishard = gt_ishard.astype(int)
        gt_hardboxes = gt_boxes[gt_ishard == 1, :]
        if gt_hardboxes.shape[0] > 0:
            # H x A
            hard_overlaps = bbox_overlaps(np.ascontiguousarray(gt_hardboxes, dtype=np.float),  # H x 4
                                          np.ascontiguousarray(anchors, dtype=np.float))  # A x 4
            hard_max_overlaps = hard_overlaps.max(axis=0)  # (A) return the value
            labels[hard_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1 # hard_max_overlaps and labels have the same shape
            max_intersec_label_inds = hard_overlaps.argmax(axis=1)  # H x 1, return the index, so the values are all less than A
            labels[max_intersec_label_inds] = -1  #

    # subsample positive labels if we have too many, less than 128
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False) # randomly clip some samples
        labels[disable_inds] = -1

    # subsample negative labels if we have too many, less than 128
    # if the num of positive samples less than 128, use negative samples to replace to ensure the total num of negative and positive samples is 256
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        # print "was %s inds, disabling %s, now %s inds" % (
        # len(bg_inds), len(disable_inds), np.sum(labels == 0))
    return labels,max_overlaps,argmax_overlaps
    
def _compute_bbox_weights(labels,A):
    bbox_inside_weights = np.zeros((A, 4), dtype=np.float32)
    # np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS) size: (4,)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS) 

    bbox_outside_weights = np.zeros((A, 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
#        num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT / (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / (np.sum(labels == 0)) + 1)
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    return bbox_inside_weights,bbox_outside_weights


def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride=[16, ], anchor_scales=[16, ]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: (1, H, W, Ax2) bg/fg scores of previous conv layer
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    gt_ishard: (G, 1), 1 or 0 indicates difficult or not
    dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
                            
    # 在feature-map上定位anchor，并加上delta，得到在实际图像中anchor的真实坐标
    # Algorithm:
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap
    """
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    A = _num_anchors = _anchors.shape[0]

    if cfg.DEBUG:
        print('anchors:')
        print(_anchors)
        print('anchor shapes:')
        print(np.hstack((_anchors[:, 2::4] - _anchors[:, 0::4],_anchors[:, 3::4] - _anchors[:, 1::4])))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    im_info = im_info[0]  # the height, width and channel of image

    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]  # feature-map's height and width

    if cfg.DEBUG:
        print('AnchorTargetLayer: height', height, 'width', width)
        print('')
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))
        print('height, width: ({}, {})'.format(height, width))
        print('rpn: gt_boxes.shape', gt_boxes.shape)
        print('rpn: gt_boxes', gt_boxes)

    all_anchors,nrof_anchors = shift_anchors(_anchors,height,width,_feat_stride,_num_anchors)
    
    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0
    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -_allowed_border) &
                           (all_anchors[:, 1] >= -_allowed_border) &
                           (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
                           (all_anchors[:, 3] < im_info[0] + _allowed_border))[0]  # height

    if cfg.DEBUG:
        print('nrof_anchors', nrof_anchors)
        print('inds_inside', len(inds_inside))

    anchors = all_anchors[inds_inside, :]# keep only inside anchors
    
    if cfg.DEBUG:
        print('anchors.shape', anchors.shape)

    labels,max_overlaps,argmax_overlaps = _compute_labels(len(inds_inside),anchors,gt_boxes, dontcare_areas, gt_ishard)

    # computing the real value of rpn-box
#    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights,bbox_outside_weights = _compute_bbox_weights(labels,len(inds_inside))

    if cfg.DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print('means:',means)
        print('stdevs:',stds)

    # map up to original set of anchors
    # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来
    labels = _unmap(labels, nrof_anchors, inds_inside, fill=-1)  # 这些anchor的label是-1，也即dontcare
    bbox_targets = _unmap(bbox_targets, nrof_anchors, inds_inside, fill=0)  # 这些anchor的真值是0，也即没有值
    bbox_inside_weights = _unmap(bbox_inside_weights, nrof_anchors, inds_inside, fill=0)  # 内部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, nrof_anchors, inds_inside, fill=0)  # 外部权重以0填充

    if cfg.DEBUG:
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print('rpn: max max_overlap', np.max(max_overlaps))
        print('rpn: num_positive', _fg_sum)
        print('rpn: num_negative', _bg_sum)
        print('rpn: num_positive avg', _fg_sum / _count)
        print('rpn: num_negative avg', _bg_sum / _count)

    rpn_labels = labels.reshape((1, height, width, A))  # reshap一下label
    rpn_bbox_targets = bbox_targets.reshape((1, height, width, A * 4))  # reshape
    rpn_bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    
  
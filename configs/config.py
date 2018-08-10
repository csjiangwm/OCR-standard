# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:35:17 2018

@author: jwm
"""

import numpy as np
from easydict import EasyDict as edict
import argparse
import keys_densenet as keys
#import keys
from utils.utils import create_dirs

def get_edict():
    __C = edict()
    cfg = __C
    __C.MODEL = 'VGGnet'
    __C.USE_GPU_NMS = True
    __C.GPU_ID = 0
    __C.ALL_GPU = False
    __C.MATCH_RATIO = 0.95
    __C.DEBUG = False
    __C.ANCHOR_SCALES = [16]
#    __C.MEAN = np.float32([102.9801, 115.9465, 122.7717])
    # MEAN=np.float32([100.0, 100.0, 100.0])
    __C.TEST_GPU_ID = 0
    __C.SCALE = 900
    __C.MAX_SCALE = 1500
    __C.TEXT_PROPOSALS_WIDTH = 0
    __C.MIN_RATIO = 0.01
    __C.LINE_MIN_SCORE = 0.6
    __C.TEXT_LINE_NMS_THRESH = 0.3
    __C.MAX_HORIZONTAL_GAP = 30
    __C.TEXT_PROPOSALS_MIN_SCORE = 0.7
    __C.TEXT_PROPOSALS_NMS_THRESH = 0.3
    __C.MIN_NUM_PROPOSALS = 0
    __C.MIN_V_OVERLAPS = 0.6
    __C.MIN_SIZE_SIM = 0.6
    __C.KEYS = keys.alphabet
    __C.EPS = 1e-14
    # Pixel mean values (BGR order) as a (1, 1, 3) array
    # We use the same pixel mean for all networks even though it's not exactly what
    # they were trained with
    __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    
    __C.TRAIN = edict()
    __C.TRAIN.HAS_RPN = True
    __C.TRAIN.WEIGHT_DECAY = 0.0005
    __C.TRAIN.RPN_PRE_NMS_TOP_N = 12000 # Number of top scoring boxes to keep before applying NMS to RPN proposals
    __C.TRAIN.RPN_POST_NMS_TOP_N = 2000 # Number of top scoring boxes to keep after applying NMS to RPN proposals
    __C.TRAIN.RPN_NMS_THRESH = 0.7 # NMS threshold used on RPN proposals
    __C.TRAIN.RPN_MIN_SIZE = 8 # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
    __C.TRAIN.RPN_CLOBBER_POSITIVES = False 
    __C.TRAIN.PRECLUDE_HARD_SAMPLES = True
    __C.TRAIN.RPN_FG_FRACTION = 0.5
    __C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
#    __C.TRAIN.SCALES_BASE = (0.25, 0.5, 1.0, 2.0, 3.0)
#    __C.TRAIN.KERNEL_SIZE = 5
#    __C.TRAIN.ASPECTS = (1, )
#    __C.TRAIN.SPATIAL_SCALE = 0.0625
    __C.TRAIN.RANDOM_DOWNSAMPLE = False
    __C.TRAIN.USE_FLIPPED = True
    __C.TRAIN.BBOX_THRESH = 0.5
    __C.TRAIN.BBOX_REG = True
    __C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    __C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    __C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
    __C.TRAIN.BBOX_NORMALIZE_TARGETS = True
    __C.TRAIN.MODEL = 'CTPN'
    __C.TRAIN.SOLVER = 'Momentum'
    __C.TRAIN.SCALES = (600, )
    __C.TRAIN.MAX_SIZE = 1000
    __C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
    __C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
    
    
    __C.TEST = edict()
    __C.TEST.NMS = 0.3
    __C.TEST.RPN_PRE_NMS_TOP_N = 12000 # Number of top scoring boxes to keep before applying NMS to RPN proposals
    __C.TEST.RPN_POST_NMS_TOP_N = 1000 # Number of top scoring boxes to keep after applying NMS to RPN proposals   
    __C.TEST.RPN_NMS_THRESH = 0.7 # NMS threshold used on RPN proposals   
    __C.TEST.RPN_MIN_SIZE = 8 # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
    __C.TEST.SCALES = (900, )
    __C.TEST.MAX_SIZE = 1500
    __C.TEST.HAS_RPN = True
    
    
#    __C.EXP_DIR = 'ctpn_end2end'
#    __C.LOG_DIR = 'ctpn'
    __C.IS_MULTISCALE = False
#    __C.NET_NAME = 'VGGnet'
#    __C.NCLASSES = 2
    __C.TRAIN.OHEM = False
#    __C.TRAIN.LOG_IMAGE_ITERS = 100
#    __C.TRAIN.DISPLAY = 10
#    __C.TRAIN.SAVE_ITERS = 100
    __C.TRAIN.EPOCH_SIZE = 10
    __C.TRAIN.LEARNING_RATE = 0.001
    __C.TRAIN.MOMENTUM = 0.9
    __C.TRAIN.GAMMA = 0.1
    __C.TRAIN.STEPSIZE = 90000
    __C.TRAIN.IMS_PER_BATCH = 1
    __C.TRAIN.RPN_BATCHSIZE = 256
    __C.TRAIN.MAX_ITERS = 500000
#    __C.TRAIN.PROPOSAL_METHOD = 'gt'
    __C.TRAIN.BG_THRESH_LO = 0.0
    __C.TRAIN.BBOX_INSIDE_WEIGHTS = [1, 1, 1, 1]
    __C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = [1, 1, 1, 1]
    __C.TRAIN.FG_FRACTION = 0.3
    __C.TRAIN.WITH_CLIP = True
    __C.TRAIN.BATCH_SIZE = 32
    __C.TRAIN.IMAGE_SIZE = (32,280)
    __C.TRAIN.LABEL_LENGTH = 10
    return cfg
    
def get_args():
    argparser = argparse.ArgumentParser(description='this is a parameter description')
    
    argparser.add_argument('--IS_TRAIN', metavar='train', default=False, type=bool,
                           help='training or testing', )
    argparser.add_argument('--CTPN_MODEL',metavar='ctpn_ckpt',default='experiments/ctpn/',type=str,
                           help='path of ctpn model for detecting text')
    argparser.add_argument('--CRNN_MODEL',metavar='crnn_model',default='experiments/crnn/model_acc97.pth',type=str,
                           help='path of crnn model for recognizing text')
    argparser.add_argument('--DENSENET_MODEL',metavar='densenet_model',default='experiments/densenet/weights_densenet.h5',
                           help='path of densenet model for recognizing text')
    argparser.add_argument('--VGG_MODEL',metavar='vgg_model',default='experiments/vgg/modelAngle.h5',type=str,
                           help='path of vgg model for detecting angle')
    argparser.add_argument('--ADJUST_ANGLE',metavar='adjustAngle',default=False, type=bool,
                           help='whether adjust the angle of image')
    argparser.add_argument('--DATA_DIR',metavar='training_dir',default='/media/jwm/DATA/work/data/VOCdevkit_CTPN',
                           help='path of training dataset')
    argparser.add_argument('--CTPN_LOGGER',metavar='ctpn_logger',default='/media/jwm/DATA/work/data/VOCdevkit_CTPN/logger',
                           help='path of CTPN training logger')
    argparser.add_argument('--PRETRAINED_MODEL',metavar='pretrained_model',
                           default='/media/jwm/DATA/work/code/project/CHINESE-OCR/ckpt_for_training/CTPN/VGG_imagenet.npy',
                           help='path of pretrained training model')
    argparser.add_argument('--SAVED_PATH',metavar='path of saved model',default='experiments/',
                           help='path of training model')
    argparser.add_argument('--DENSENET_LOGGER',metavar='ctpn_logger',default='experiments/densenet_logger',
                           help='path of CTPN training logger')
    args = argparser.parse_args()
    return args
    
    
def process_config():
    args = get_args()
    config = get_edict()
    config.update(vars(args))
    create_dirs([args.CTPN_MODEL,args.CTPN_LOGGER,args.DENSENET_LOGGER,args.SAVED_PATH])
    return edict(config)
    
cfg = process_config()

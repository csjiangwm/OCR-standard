# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:12:29 2018

@author: jwm
"""

from configs.config import cfg
from models.DenseNet import DenseNet
from data_loader.text_generator import TextGenerator
from trainers.densenet_trainer import DenseNetTrainer

cfg.IS_TRAIN = True

def train():
    '''
       --DATA_DIR /media/jwm/DATA/work/data/CRNN
       --PRETRAINED_MODEL /home/jwm/Desktop/OCR-standard/experiments/densenet/weights_densenet.h5
       --SAVED_PATH experiments/densenet_ckpt
       --DENSENET_LOGGER experiments/densenet_logger
    '''
    data = TextGenerator(cfg)
    model = DenseNet(cfg, data.nrof_classes)
    trainer = DenseNetTrainer(cfg, data, model)
    trainer.train()

if __name__ == '__main__':
    train()
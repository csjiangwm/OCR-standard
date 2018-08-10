# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 09:35:35 2018

@author: jwm
"""

from configs.config import cfg
from utils.utils import get_sess
from models.CTPN import CTPN
from data_loader.data_loader import DataLoader
from trainers.ctpn_trainer import CTPNTrainer
from loggers.logger import Logger

cfg.IS_TRAIN = True

def train():
    model = CTPN(cfg)
    data = DataLoader(cfg)
#    imdb = data.load_imdb('voc_2007_trainval')
#    roidb = data.get_training_roidb(imdb)
    sess = get_sess()
    logger = Logger(sess,cfg)
    trainer = CTPNTrainer(sess, model, data, logger)
    print('Solving...')
    trainer.train(cfg.TRAIN.MAX_ITERS, restore=False)
    print('done solving')

if __name__ == '__main__':
    train()
    
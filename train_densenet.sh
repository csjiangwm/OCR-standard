#!/bin/sh

python DenseNet_train.py --DATA_DIR /media/jwm/DATA/work/data/CRNN --PRETRAINED_MODEL /home/jwm/Desktop/OCR-standard/experiments/densenet/weights_densenet.h5 --SAVED_PATH experiments/densenet_ckpt --DENSENET_LOGGER experiments/densenet_logger


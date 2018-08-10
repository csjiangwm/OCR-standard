# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:23:29 2018

@author: jwm
"""

import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

#class resizeNormalize(object):
#    def __init__(self, size, interpolation=Image.BILINEAR):
#        self.size = size
#        self.interpolation = interpolation
#        self.toTensor = transforms.ToTensor()
#
#    def __call__(self, img):
#        img = img.resize(self.size, self.interpolation)
#        img = self.toTensor(img)
#        img.sub_(0.5).div_(0.5)
#        return img

def data_parallel(model, _input, ngpu):
    if isinstance(_input.data, torch.cuda.FloatTensor) and ngpu > 1:
        output = nn.parallel.data_parallel(model, _input, range(ngpu))
    else:
        output = model(_input)
    return output
    
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, ngpu):
        super(BidirectionalLSTM, self).__init__()
        self.ngpu = ngpu
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, _input):
        recurrent, _ = data_parallel(self.rnn, _input, self.ngpu)  # [T, N, hidden * 2 (because bilstm)]
        T, b, h = recurrent.size() # T is W, b is N
#        print recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = data_parallel(self.embedding, t_rec, self.ngpu)  # [T * N, nOut]
        output = output.view(T, b, -1)
        return output
        
class CRNN(nn.Module):
    def __init__(self, config, imgH, nc, nclass, nh, ngpu, n_rnn=2, leakyRelu=False):
        '''parameters
           config:
           imgH: height of input image
           nc:   number of channels
           nclass: number of classes
           nh:   number of hidden layer cells for lstm
        '''
        super(CRNN, self).__init__()
        self.config = config
        self.ngpu = ngpu
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2] # kernal size
        ps = [1, 1, 1, 1, 1, 1, 0] # padding size
        ss = [1, 1, 1, 1, 1, 1, 1] # stride size
        nm = [64, 128, 256, 256, 512, 512, 512] # channel size

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),nn.LeakyReLU(0.2, inplace=True)) 
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        # 在宽方向上stide=1， padding=1
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh, ngpu),
                                 BidirectionalLSTM(nh, nh, nclass, ngpu))
            
    def forward(self, _input):
        # cnn(input) 相当于将input放入cnn的nn.Sequential()中依次执行每个module，每个module类内有个forward
        conv = data_parallel(self.cnn, _input, self.ngpu)# conv features
        b, c, h, w = conv.size()
#        print conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # [N, W, C]
        conv = conv.permute(2, 0, 1)  # [W, N, C]
        # rnn(input) 相当于将input放入rnn的nn.Sequential()中依次执行每个module
        output = data_parallel(self.rnn, conv, self.ngpu)# rnn features
        return output
        
    def recognize(self,image,data):
        scale = image.size[1] * 1.0 / 32
        w = int(image.size[0] / scale)
        # print "im size:{},{}".format(image.size,w)
#        transformer = resizeNormalize((w, 32))
        
        def transformer(img):
            img = img.resize((w, 32), Image.BILINEAR)
            img = transforms.ToTensor()(img)
            img.sub_(0.5).div_(0.5) # each element subtract 0.5 and then divide 0.5
            return img
        
        if torch.cuda.is_available() and self.config.ALL_GPU:
            image = transformer(image).cuda()
        else:
            image = transformer(image).cpu() # CHW
            
        image = image.view(1, *image.size()) # NCHW
        image = Variable(image)
#        self.eval()
#        print image.size()
        preds = self.forward(image) # call forward
        values, preds = preds.max(2) # dim=2
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = data.decode_crnn(preds.data, preds_size.data, raw=False)
        if len(sim_pred) > 0:
            if sim_pred[0] == u'-':
                sim_pred = sim_pred[1:]
    
        return sim_pred
        
# -*- coding: utf-8 -*-
'''
Created on Tue Jul 17 17:33:25 2018

@author: jwm
'''

import os
import numpy as np
import scipy.sparse
import cPickle as pickle
import xml.etree.ElementTree as ET
import PIL

from configs.config import cfg


classes = ('__background__','text')
nrof_classes = len(classes)

class pascal_voc():
    def __init__(self, image_set, year, devkit_path=None):
        self._image_set = image_set
        self.name = 'voc_' + year + '_' + image_set
        self._devkit_path = cfg.DATA_DIR if devkit_path is None else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + year)

        self._class_to_ind = dict(list(zip(classes, list(range(nrof_classes)))))
        self.image_index = self._load_image_set_index()
        self.get_roidb()

        assert os.path.exists(self._devkit_path), 'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def get_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        self.roidb = gt_roidb

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, nrof_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            
            difficult = obj.find('difficult')
            ishards[ix] = 0 if difficult == None else int(difficult.text)

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            # cls = self._class_to_ind[obj.find('name').text]

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_ishard': ishards,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas
        }
        
    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path
        
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
        
    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0] for i in range(self.num_images)]
        
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])
        
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages', index + self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path
        
    def append_flipped_images(self):
        nrof_images = len(self.image_index)
        widths = self._get_widths()
        for i in range(nrof_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            for b in range(len(boxes)):
                if boxes[b][2] < boxes[b][0]:
                    boxes[b][0] = 0
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {
                'boxes': boxes,
                'gt_overlaps': self.roidb[i]['gt_overlaps'],
                'gt_classes': self.roidb[i]['gt_classes'],
                'flipped': True
            }

            if 'gt_ishard' in self.roidb[i] and 'dontcare_areas' in self.roidb[i]:
                entry['gt_ishard'] = self.roidb[i]['gt_ishard'].copy()
                dontcare_areas = self.roidb[i]['dontcare_areas'].copy()
                oldx1 = dontcare_areas[:, 0].copy()
                oldx2 = dontcare_areas[:, 2].copy()
                dontcare_areas[:, 0] = widths[i] - oldx2 - 1
                dontcare_areas[:, 2] = widths[i] - oldx1 - 1
                entry['dontcare_areas'] = dontcare_areas

            self.roidb.append(entry)

        self.image_index = self.image_index * 2
        
    



if __name__ == '__main__':
    d = pascal_voc('trainval', '2007')

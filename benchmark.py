#%%
"""
  This is the benchmark for mtcnn-pytorch project, taking 2 dataset:
    1. the WIDER FACE validation datasets;
    2. the self maintained production environment images;
  Author: Sherk;
  First drafted : 2019-6-27;
"""
import sys

import os
from os.path import join, split
from glob import glob
from tqdm import tqdm
import re

import numpy as np
import numpy.random as npr
import pandas as pd

import torch
import cv2

from scripts.MTCNN import MTCNN, LoadWeights
from scripts.Nets import PNet, RNet, ONet
from scripts.util.utility import iou, boxes_extract


USE_CUDA = True
GPU_ID = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in GPU_ID])
device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")


prefix = '20190624'


# pnet
pnet_weight_path = "scripts/models/pnet_{}_final.pkl".format(prefix)
pnet = PNet(test=True)
LoadWeights(pnet_weight_path, pnet)
pnet.to(device)

# rnet
rnet_weight_path = "scripts/models/rnet_{}_final.pkl".format(prefix)
rnet = RNet(test=True)
LoadWeights(rnet_weight_path, rnet)
rnet.to(device)

# onet
onet_weight_path = "scripts/models/onet_{}_final.pkl".format(prefix)
onet = ONet(test=True)
LoadWeights(onet_weight_path, onet)
onet.to(device)

mtcnn = MTCNN(
  detectors=[pnet, rnet, onet],
  device=device,
  min_face_size=20,
  threshold=[0.6, 0.7, 0.7],
  scalor=0.79)

#%%
filenames = os.listdir('img')

missing_detection = 0
false_detection = 0
all_detection = 0
all_labels = 0

for filename in tqdm(filenames):
  iou_threshold = 0.4

  image = 'img/{}'.format(filename)
  img = cv2.imread(image)

  boxes_det = mtcnn.detect(img)
  # boxes_det = boxes_det[boxes_det[:, 2] >= 8]
  # boxes_det = boxes_det[boxes_det[:, 3] >= 8]
  if boxes_det is not None:
    boxes_det[:, 2] += boxes_det[:, 0]
    boxes_det[:, 3] += boxes_det[:, 1]

  xml_file = 'anno/{}.xml'.format( '.'.join( filename.split('.')[:-1] ) )
  boxes_lab = boxes_extract(xml_file)
  # boxes_lab = boxes_lab[boxes_lab[:, 3] - boxes_lab[:, 1] >= 8]
  # boxes_lab = boxes_lab[boxes_lab[:, 2] - boxes_lab[:, 0] >= 8]

  # ===================================================================
  if boxes_lab is None:
    if boxes_det is None:
      continue
    else:
      false_detection += len(boxes_det)
      continue
  
  if boxes_det is None:
    if boxes_lab is None:
      continue
    else:
      missing_detection += len(boxes_lab)
      continue

  for box in boxes_lab:
    if max(iou(box, boxes_det)) < iou_threshold:
      missing_detection += 1
      # Blue stands for missings
      cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    # Green is from label
    # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
  
  for box in boxes_det:
    if max(iou(box, boxes_lab)) < iou_threshold:
      false_detection += 1
      # red stands for false detection
      cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    # Red is from detector
    # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

  cv2.imwrite('output/{}'.format(filename), img)
  all_detection += len(boxes_det)
  all_labels += len(boxes_lab)

print('Detect\tMissing\tAll\tFalse')
print('{}\t{}\t{}\t{}'.format(all_detection,
  missing_detection, all_labels, false_detection))

precision = round(1 - false_detection / (all_labels + false_detection), 4)
print('Precision: {}'.format(precision))

recall = round(1 - missing_detection / (all_detection + missing_detection), 4)
print('Recall: {}'.format(recall))











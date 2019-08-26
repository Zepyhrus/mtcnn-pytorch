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
from scripts.util.utility import iou, boxes_extract, parse_args

test_type = parse_args(sys.argv[1:]).test_type

print('Doing {} benchmark...'.format(test_type))

USE_CUDA = True
GPU_ID = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in GPU_ID])
device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")


prefix = '20190822'


# pnet
pnet_weight_path = "scripts/models/pnet_{}_final.pkl".format(prefix)
pnet = PNet(test=True)
LoadWeights(pnet_weight_path, pnet)

# save the model to pt
pnet.eval()
dummy_tensor = torch.rand(1, 3, 12, 12)
script_model = torch.jit.trace(pnet, dummy_tensor)

torch.jit.save(script_model, 'pnet.new.pt')

pnet.to(device)


# rnet
rnet_weight_path = "scripts/models/rnet_{}_final.pkl".format(prefix)
rnet = RNet(test=True)
LoadWeights(rnet_weight_path, rnet)

rnet.eval()
dummy_tensor = torch.rand(1, 3, 24, 24)
script_model = torch.jit.trace(rnet, dummy_tensor)

torch.jit.save(script_model, 'rnet.new.pt')

rnet.to(device)

# onet
onet_weight_path = "scripts/models/onet_{}_final.pkl".format(prefix)
onet = ONet(test=True)
LoadWeights(onet_weight_path, onet)

onet.eval()
dummy_tensor = torch.rand(1, 3, 48, 48)
script_model = torch.jit.trace(onet, dummy_tensor)

torch.jit.save(script_model, 'onet.new.pt')

onet.to(device)

# initialize the mtcnn model
mtcnn = MTCNN(
  detectors=[pnet, rnet, onet],
  device=device,
  min_face_size=40,
  threshold=[0.7, 0.8, 0.9],
  scalor=0.79)

#%% benchmark for production 373 images
if test_type == 'accuracy':
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
      if boxes_det is not None:
        false_detection += len(boxes_det)
        all_detection += len(boxes_det)
      continue

    if boxes_det is None:
      if boxes_lab is not None:
        missing_detection += len(boxes_lab)
        all_labels += len(boxes_lab)
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

  print('Detect\tMissing\tFalse\tAll')
  print('{}\t{}\t{}\t{}'.format(all_detection,
    missing_detection, false_detection, all_labels))

  precision = 1 - false_detection / all_detection
  print('Precision: {}'.format(round(precision, 4)))

  recall = 1 - missing_detection / all_labels
  print('Recall: {}'.format(round(recall, 4)))

  f1_score = 2 * precision * recall / (precision + recall)
  print('F1 score: {}'.format(round(f1_score, 4)))



#%% speed benchmark
# taking images benchmark from folder benchmark
if test_type == 'speed':
  import timeit

  images = glob('benchmark/*')

  for image in images:
    img = cv2.imread(image)

    boxes = mtcnn.detect(img)

    if boxes is not None:
      for box in boxes:
        cv2.rectangle(img, (box[0], box[1]),
          (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)
    
    cv2.imwrite(image.replace('benchmark', 'output'), img)

    print(round(timeit.timeit(lambda: mtcnn.detect(img), number=10)*100), end='\t')
    print('{} faces detected'.format(len(boxes)), end='\t')
    print(image)





#%%
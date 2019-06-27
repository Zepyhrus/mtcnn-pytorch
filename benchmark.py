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

import numpy as np
import numpy.random as npr
import pandas as pd

import torch
import cv2

from scripts.MTCNN import MTCNN, LoadWeights
from scripts.Nets import PNet, RNet, ONet


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
  scalor=0.91)



images = glob('/home/ubuntu/Workspace/dataset/benchmark/*')
# image = 'img/faces2.jpg'
for image in tqdm(images):
  img  = cv2.imread(image)
  image_name = split(image)[-1]

  # print(img.shape)
  bboxes = mtcnn.detect(img)

  if bboxes is not None:
    for b in bboxes:
      x, y, w, h = [int(z) for z in b[0:4]]
      img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
      cv2.imwrite("output/{}".format(image_name), img)
  else:
    print(image_name)




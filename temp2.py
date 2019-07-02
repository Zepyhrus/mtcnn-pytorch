#
import sys
import os
from os.path import join, split

import xml.etree.ElementTree as et
import numpy as np
import numpy.random as npr
import pandas as pd


# import cv2
#
# root = et.parse('anno/sls/faces1.xml')
# --------x1_---y1_----x2_---y2_--------
boxes = [[68,    32,   120,  144],
         [125,  162,   176,  200],
         [321,  212,   352,  260]]
# --------x1----y1-----x2----y2---------
# box =  [127,  167,   177,  212]

boxes_det = [[127, 167, 177, 212], [366, 334, 370, 351]]


# box = boxes_det[0]
for box in boxes_det:
  print('-'*82)
  x1, y1, x2, y2 = box  # 127, 167, 177, 212
  boxes = np.asarray(boxes)
  x1_, y1_, x2_, y2_ = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

  boxes[:, 0] -= x2
  boxes[:, 2] -= x1

  print(boxes)

  

  # wd = max(min(abs(x2 - x1_), abs(x2_ - x1)) * np.sign(x2 - x1_) * np.sign(x2_ - x1), 0)
  # ht = max(min(abs(y2 - y1_), abs(y2_ - y1)) * np.sign(y2 - y1_) * np.sign(y2_ - y1), 0)
  # print(wd, ht)
  
  # intersection = wd * ht
  # union = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intersection

  # iou_ = intersection / union
  # print(iou_)





def iou(box, boxes):

  pass

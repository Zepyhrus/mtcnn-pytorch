"""
  This is the playground for mtcnn-pytorch project
"""
import sys

import os
from os.path import join, split
from glob import glob


IMAGE_PATH = '/home/ubuntu/Workspace/dataset/WIDER_FACE'
TRAIN_PATH = join(IMAGE_PATH, 'WIDER_train/images')
VAL_PATH = join(IMAGE_PATH, 'WIDER_val/images')

# merge WIDER FACE train and val data sets
sub_folders = os.listdir(TRAIN_PATH)

cum = 0
for folder in sub_folders:
  cum += len(os.listdir(join(TRAIN_PATH, folder)))

print(cum)

"""
  Using:
    cat wider_face_train_bbx_gt.txt | wc -l
      : yielding 185184 lines of record
    cat wider_face_val_bbx_gt.txt >> wider_face_train_bbx_gt.txt 
    cat wider_face_train_bbx_gt.txt | wc -l
      : yielding 231344 lines of record
"""


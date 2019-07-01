# coding: utf-8
import sys
sys.path.append(sys.path[0] + "/../")

from datetime import datetime


from data.DataSouce import DataSource
from data.augmentation import *
import os
import random
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from util.torchutil import SaveCheckPoint
from util.Logger import Logger
from util.utility import parse_args
from Nets import *


if not os.path.exists("./log/"):
  os.mkdir("./log/")
log = Logger("./log/{}_{}.log".format(__file__.split('/')[-1],
                       time.strftime("%Y%m%d-%H%M%S"), time.localtime), level='debug').logger

USE_CUDA = True
GPU_ID = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in GPU_ID])
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

pre_checkpoint = None
resume = False

train_batch = 400
display = 100

base_lr = 0.001
clip_grad = 120.0
momentum = 0.9
gamma = 0.1
weight_decay = 0.0005
save_interval = 25000

#TODO: modify the prefix to train different net
net_type = parse_args(sys.argv[1:]).net_type
assert net_type in ['PNET', 'RNET', 'ONET'], 'Wrong net type'

if net_type == 'PNET':
  prefix = "p"
  INPUT_IMAGE_SIZE = 12
  stepsize = [50000, 80000, 100000, 120000, 140000]
  max_iter = 150000
  train_ratio = 8
  loss_ratio = (3, 1)
  net = PNet()
elif net_type == 'RNET':
  prefix = 'r'
  INPUT_IMAGE_SIZE = 24
  stepsize = [80000, 140000, 200000, 250000]
  max_iter = 300000
  train_ratio = 2
  loss_ratio = (1, 1)
  net = RNet()
elif net_type == 'ONET':
  prefix = 'o'
  INPUT_IMAGE_SIZE = 48
  stepsize = [30000, 50000, 60000, 70000]
  max_iter = 80000
  train_ratio = 1
  loss_ratio = (1, 3)
  net = ONet()




save_dir = "./models"

if not os.path.exists(save_dir):
  os.mkdir(save_dir)

surfix = datetime.now().strftime('%Y%m%d')
save_prefix = save_dir + "/{}net_{}".format(prefix, surfix)


root_dir = r"../dataset/"


topk = 0.7
MEANS = [127.5, 127.5, 127.5]
train_anno_path = []
val_anno_path = []

train_anno_path += [os.path.join(root_dir, "train_faces_{}/pos/image_pos".format(prefix))]
train_anno_path += [os.path.join(root_dir, "train_faces_{}/pos/label_pos".format(prefix))]

train_anno_path += [os.path.join(root_dir, "train_faces_{}/part/image_part".format(prefix))]
train_anno_path += [os.path.join(root_dir, "train_faces_{}/part/label_part".format(prefix))]

train_anno_path += [os.path.join(root_dir, "train_faces_{}/neg/image_neg".format(prefix))]
train_anno_path += [os.path.join(root_dir, "train_faces_{}/neg/label_neg".format(prefix))]



def train(net):
  start_epoch = 0
  # dataset
  train_dataset = DataSource(train_anno_path, transform=Compose([
    RandomMirror(0.5), SubtractFloatMeans(MEANS), ToPercentCoords(), PermuteCHW()
  ]), ratio=train_ratio, image_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))

  # net
  

  # optimizer and scheduler
  # Modified by Sherk, Adam optimizer is applied for faster convergence
  optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
  # optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=weight_decay)  # Adam takes no momentum
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, stepsize, gamma)

  # device
  if USE_CUDA:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
  if pre_checkpoint:
    cp = torch.load(pre_checkpoint)
    net.load_state_dict(cp['weights'])
    log.info("=> load state dict from {}...".format(pre_checkpoint))
    if resume:
      optimizer.load_state_dict(cp['optimizer'])
      scheduler.load_state_dict(cp['scheduler'])
      start_epoch = cp['epoch']
      log.info("=> resume from epoch: {}, now the lr is: {}".format(start_epoch, optimizer.param_groups[0]['lr']))

  net.to(device)

  k = 0
  for epoch in range(start_epoch, max_iter + 1):
    net.train()
    images, targets = train_dataset.getbatch(train_batch)
    images = images.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()

    pred_cls, pred_bbox = net(images)

    loss_cls = AddClsLoss(pred_cls, targets, topk)
    loss_reg = AddRegLoss(pred_bbox, targets)
    loss = loss_ratio[0] * loss_cls + loss_ratio[1] * loss_reg

    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)

    optimizer.step()
    scheduler.step()

    if k % display == 0:
      acc_cls = AddClsAccuracy(pred_cls, targets)
      acc_reg = AddBoxMap(pred_bbox, targets, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

      log.info("train iter: {}, lr: {}, loss: {:.4f}, cls loss: {:.4f}, bbox loss: {:.4f}, cls acc: {:.4f}, bbox acc: {:.4f}".format(
        k, optimizer.param_groups[0]['lr'], loss.item(), loss_cls.item(), loss_reg.item(), acc_cls, acc_reg))

    # donot save the intermediate .pkls
    # if k % save_interval == 0:
    #   path = save_prefix + "_iter_{}.pkl".format(k)
    #   SaveCheckPoint(path, net, optimizer, scheduler, epoch)
    #   log.info("=> save model: {}".format(path))

    k += 1

  log.info("optimize done...")
  path = save_prefix + "_final.pkl"
  SaveCheckPoint(path, net, optimizer, scheduler, max_iter)
  log.info("=> save model: {} ...".format(path))


if __name__ == '__main__':
  train(net)

  if net_type == 'ONET':
    print('='*35 + '  Finished!  ' + '='*35)
    

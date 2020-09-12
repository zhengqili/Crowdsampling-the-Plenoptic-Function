from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.train_options import TrainOptions
import sys
from data.data_loader import *
from models.models import create_model
import random
import models
torch.manual_seed(0)
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch


if opt.dataset == 'trevi':
    scene_id = 36
elif opt.dataset == 'pantheon':
    scene_id = 23
elif opt.dataset == 'coeur':
    scene_id = 13
elif opt.dataset == 'rock':
    scene_id = 11
elif opt.dataset == 'navona':
    scene_id = 57

root = '/home/zl548/'
data_dir = root + '/Flickr100M/%04d/dense_sphere_0/'%scene_id
local_dir = root + '/Crowdsampling-the-Plenoptic-Function/'

opt.root = root
opt.local_dir = local_dir
opt.root = root

test_num_threads = 3
test_data_loader = CreateLandmarksDataLoader(opt, data_dir, 'test', test_num_threads)
test_dataset = test_data_loader.load_data()
test_data_size = len(test_data_loader)
print('========================= %s test #images = %d ========='%(opt.dataset, test_data_size))


model = create_model(opt, _isTrain=False)
model.switch_to_eval()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
global_step = 0

total_l1 = 0.
total_psnr = 0.
total_lpips = 0.
count = 0.

lpips_model = models.PerceptualLoss(model='net', net='alex', use_gpu=True, gpu_ids=[1])

for i, data in enumerate(test_dataset):

    global_step = global_step + 1
    print('global_step', global_step)
    targets = data

    model.set_input(targets)
    l1_error, psnr, lpips, valid = model.evaluate_all(lpips_model)

    print('l1_error ', l1_error)
    print('psnr ', psnr)
    print('lpips ', lpips)

    total_l1 += l1_error
    total_psnr += psnr
    total_lpips += lpips
    count += valid

avg_l1 = total_l1/count
avg_psnr = total_psnr/count
avg_lpips = total_lpips/count


print('avg_l1 ', avg_l1)
print('avg_psnr ', avg_psnr)
print('avg_lpips ', avg_lpips)


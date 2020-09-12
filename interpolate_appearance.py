from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.test_options import TestOptions
import sys
from data.data_loader import *
from models.models import create_model
import random

import json
torch.manual_seed(0)

opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

root = '/home/zl548/'
local_dir = root + '/Crowdsampling-the-Plenoptic-Function/'

opt.root = root
opt.local_dir = local_dir


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

data_dir = root + '/Flickr100M/%04d/dense_sphere_0/'%scene_id

test_num_threads = 3
test_data_loader = CreateLandmarksDataLoader(opt, data_dir, 'interpolation', test_num_threads,
											  img_a_name=opt.img_a_name, img_b_name=opt.img_b_name, img_c_name=opt.img_c_name)

test_dataset = test_data_loader.load_data()
train_data_size = len(test_data_loader)
print('========================= %s training #images = %d ========='%(opt.dataset,train_data_size))


model = create_model(opt, _isTrain=False)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
global_step = 0

save_root_dir = './viz_interpolation_%s/'%opt.dataset

if not os.path.exists(save_root_dir):
     os.makedirs(save_root_dir)

for i, data in enumerate(test_dataset):
	targets = data
	model.set_input(targets)
	model.interpolate_appearance(save_root_dir)
	sys.exit()

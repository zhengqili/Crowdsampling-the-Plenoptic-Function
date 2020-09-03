# from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
import modules
# from torchvision import utils

# import senet
# import resnet
# import densenet
imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):
        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)

    def forward(self, x):
        x_normalized_r = (x[:, 0, :, :] - imagenet_stats['mean'][0])/imagenet_stats['std'][0]
        x_normalized_g = (x[:, 1, :, :] - imagenet_stats['mean'][1])/imagenet_stats['std'][1]
        x_normalized_b = (x[:, 2, :, :] - imagenet_stats['mean'][2])/imagenet_stats['std'][2]

        x_normalized = torch.cat((x_normalized_r.unsqueeze(1), x_normalized_g.unsqueeze(1), x_normalized_b.unsqueeze(1)), 1)

        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x_normalized)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)

        # print('x_decoder ', x_decoder.size())

        x_mff = self.MFF(x_block0, x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2),x_decoder.size(3)])
        
        # print('x_mff ', x_mff.size())

        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out
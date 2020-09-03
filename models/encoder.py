"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.base_network import BaseNetwork

from models.normalization import get_nonspade_norm_layer
import sys
from . import networks

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt, num_inputs):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.nef
        norm_layer = get_nonspade_norm_layer(opt, 'spectralinstance')
        self.layer1 = norm_layer(nn.Conv2d(num_inputs, ndf, kw, stride=2, padding=pw)) 
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.fc = nn.Conv2d(ndf * 8, 8, 1)
        # self.fc_mu = nn.Linear(ndf * 8, opt.num_latent_f)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt


    def forward(self, x):
        # if x.size(2) != 256 or x.size(3) != 256:
        # x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.layer6(self.actvn(x))
        # x = self.layer7(self.actvn(x))

        x = self.fc(self.actvn(x))
        return x


class AppearanceEncoder(nn.Module):

    def __init__(self, opt):
        super(AppearanceEncoder, self).__init__()

        # Define encoder type
        self.style_encoder = networks.StyleEncoder(n_downsample=4, input_dim=3, 
                                                   dim=opt.nef, style_dim=opt.num_latent_f, 
                                                   norm='none', activ='lrelu', pad_type='reflect')
        self.local_encoder = ConvEncoder(opt, num_inputs=3+3+opt.num_mpi_f)

        # self.final_conv = nn.Conv2d(512+16, opt.num_latent_f, 1, 1, 0)
        self.final_fc = nn.Linear(512+256, opt.num_latent_f)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x, z):
        latent_z_style = self.style_encoder(z)
        latent_z_local = self.local_encoder(x)

        latent_z_concat = torch.cat([latent_z_style, latent_z_local.view(1, 512, 1, 1)], dim=1)

        return self.final_fc(latent_z_concat.view(latent_z_concat.size(0), -1)).unsqueeze(-1).unsqueeze(-1)
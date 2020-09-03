from __future__ import division
import torch
import torch.nn as nn
import torch.sparse
from torch.autograd import Variable
import numpy as np
import sys
from torch.autograd import Function
import math
# from scipy import misc 
# import scipy
import functools
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import scipy.io as sio
###############################################################################
# Functions
###############################################################################
VERSION = 4
EPSILON = 1e-8




class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim=3):
        super(MsImageDis, self).__init__()
        self.n_layer = 4 # params['n_layer']
        self.gan_type = 'lsgan' #params['gan_type']
        self.dim = 64 #params['dim']
        self.norm = 'none' #params['norm']
        self.activ = 'lrelu' #params['activ']
        self.num_scales = 3 #params['num_scales']
        self.pad_type = 'reflect' #params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real, gt_mask_0, gt_mask_1):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):

            gt_mask_0_x = nn.functional.interpolate(gt_mask_0, size=[out0.size(2), out0.size(3)], mode='nearest')
            gt_mask_1_x = nn.functional.interpolate(gt_mask_1, size=[out0.size(2), out0.size(3)], mode='nearest')

            num_valid_0 = torch.sum(gt_mask_0_x) + 1e-8
            num_valid_1 = torch.sum(gt_mask_1_x) + 1e-8

            if self.gan_type == 'lsgan':
                loss += torch.sum( ((out0 - 0.) * gt_mask_0_x)**2 )/num_valid_0 + torch.sum( ((out1 - 1.) * gt_mask_1_x)**2 )/num_valid_1
                # loss += (torch.mean((out0 - 0) * **2) + torch.mean((out1 - 1)**2))/2.
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake, gt_mask):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            gt_mask_x = nn.functional.interpolate(gt_mask, size=[out0.size(2), out0.size(3)], mode='nearest')

            num_valid = torch.sum(gt_mask_x) + 1e-8

            if self.gan_type == 'lsgan':
                loss += torch.sum( ((out0 - 1) * gt_mask_x)**2 )/num_valid
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss



class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epoch, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)    
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm)    
    elif norm_type == 'pixel':
        norm_layer = functools.partial(PixelNorm)# PixelNorm
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # if len(gpu_ids) > 0:f
        # assert(torch.cuda.is_available())
        # net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)

def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


class padding(nn.Module):
    def __init__(self):
        super(padding, self).__init__()
        self.wpad = nn.ReplicationPad2d((0, -1, 0, 0))
        self.hpad = nn.ReplicationPad2d((0, 0, 0, -1))
        
    def forward(self, input, targetsize):
        if input.size()[2] != targetsize[2]:
            input = self.hpad(input)
        if input.size()[3] != targetsize[3]:
            input = self.wpad(input)
        return input


def define_G(input_nc, output_nc, nz, ngf, norm='instance', nl='relu',
             use_dropout=False, init_type='xavier', init_gain=0.02,
             where_add='input', upsample='bilinear', style_dim=8):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, num_downs=5, ngf=ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample)

    elif where_add == 'adain':
        net = G_Unet_AdaIN(input_nc, output_nc, nz, num_downs=5, ngf=ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample, style_dim=style_dim)

    elif where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, num_downs=5, ngf=ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, upsample=upsample)
    elif where_add == 'middle':
        net = G_Unet_add_middle(input_nc, output_nc, nz, num_downs=5, ngf=ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                use_dropout=use_dropout, upsample=upsample)
    elif where_add == 'shallow':
        net = G_Unet_add_input_shallow(input_nc, output_nc, nz, num_downs=5, ngf=ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                       use_dropout=use_dropout, upsample=upsample)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain)


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        # self.model += [nn.Conv2d(dim, 16, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)



class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)        



class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=32,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='basic', ref_feature_mpi=None):
        super(G_Unet_add_input, self).__init__()
        self.nz = nz
        max_nchn = 8

        # construct unet structure
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        # for i in range(num_downs - 5):
            # unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   # norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        print('input_nc + nz ', input_nc + nz)

        unet_block = UnetBlock(input_nc + nz, output_nc, ngf, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block
        self.ref_feature_mpi = ref_feature_mpi
    
    def forward(self, x):

        # x_with_z = x  # no z

        # print('x_with_z ', x_with_z.size())

        return self.model(x)


class G_Unet_AdaIN(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=32,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='bilinear', style_dim=8):
        super(G_Unet_AdaIN, self).__init__()
        self.nz = nz
        max_nchn = 8
        self.style_dim = style_dim

        # construct unet structure
        unet_block = UnetBlockAdaIN(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, style_dim=style_dim)
        # for i in range(num_downs - 5):
            # unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   # norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlockAdaIN(ngf * 4, ngf * 4, ngf * max_nchn, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, style_dim=style_dim)
        unet_block = UnetBlockAdaIN(ngf * 2, ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, style_dim=style_dim)
        unet_block = UnetBlockAdaIN(ngf, ngf, ngf * 2, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, style_dim=style_dim)

        # print('input_nc + nz ', input_nc + nz)

        unet_block = UnetBlockAdaIN(input_nc + nz, output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, style_dim=style_dim)

        self.model = unet_block
        # self.ref_feature_mpi = ref_feature_mpi
    
    def forward(self, x):
        input_buffer, style_code = x[:, 0:-self.style_dim, :, :], x[0:1, -self.style_dim:, 0, 0]

        # print('input_buffer ', input_buffer.size())
        # print('style_code ', style_code.size())
        # sys.exit()

        return self.model(input_buffer, style_code)




class G_Unet_add_input_shallow(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=32,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='basic', ref_feature_mpi=None):
        super(G_Unet_add_input_shallow, self).__init__()
        self.nz = nz
        max_nchn = 4

        # construct unet structure
        unet_block = UnetBlockShallow(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                                      innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        
        unet_block = UnetBlockShallow(ngf * 4, ngf * 4, ngf * max_nchn, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlockShallow(ngf * 2, ngf * 2, ngf * 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlockShallow(ngf, ngf, ngf * 2, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        unet_block = UnetBlockShallow(input_nc + nz, output_nc, ngf, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block
        self.ref_feature_mpi = ref_feature_mpi
    
    def forward(self, x, z=None):
        x_with_z = x  # no z

        # print('x_with_z ', x_with_z.size())

        return self.model(x_with_z)


class G_Unet_add_middle(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='basic', ref_feature_mpi=None):
        super(G_Unet_add_middle, self).__init__()
        self.nz = nz
        max_nchn = 8

        # construct unet structure
        unet_block = UnetBlock_z_middle(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                                        nz=nz, submodule=None, innermost=True, 
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock_z_middle(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, 
                                            nz=0, submodule=unet_block,
                                            norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_z_middle(ngf * 4, ngf * 4, ngf * max_nchn, 
                                        nz=0, submodule=unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_z_middle(ngf * 2, ngf * 2, ngf * 4,
                                        nz=0, submodule=unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_z_middle(ngf, ngf, ngf * 2,
                                        nz=0, submodule=unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        unet_block = UnetBlock_z_middle(input_nc, output_nc, ngf,
                                        nz=0, submodule=unet_block,
                                        outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block
        self.ref_feature_mpi = ref_feature_mpi
    
    def forward(self, x, z=None):
        return self.model(x, z)



# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='bilinear'):
        super(G_Unet_add_all, self).__init__()
        self.nz = nz
        # construct unet structure
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                          norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block,
                                      outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='replicate'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.pad = padding()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=3, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)

        # num_in_groups = inner_nc//16
        # num_out_groups = outer_nc//16

        downnorm = norm_layer() if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer() if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):

        if self.outermost:
            return (self.model(x) + 1.)/2.
        else:
            x_out = self.model(x)
            x_out = self.pad(x_out, x.size())

            return torch.cat([x_out, x], 1)


class UnetBlockAdaIN(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='replicate', style_dim=8):
        super(UnetBlockAdaIN, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.pad = padding()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=3, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)

        # num_in_groups = inner_nc//16
        # num_out_groups = outer_nc//16


        print('norm_layer ', norm_layer)

        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()

        self.upnorm = AdaptiveInstanceNorm(outer_nc, style_dim) #norm_layer() if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            # model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            # if upnorm is not None:
                # up += [upnorm]
            # model = down + up
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            # if upnorm is not None:
                # up += [upnorm]

            # if use_dropout:
                # model = down + [submodule] + up + [nn.Dropout(0.5)]
            # else:
            # model = down + [submodule] + up

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

        # self.model = nn.Sequential(*model)

    def forward(self, x, z):

        if self.outermost:
            x1 = self.down(x)
            x2 = self.submodule(x1, z)
            return (self.up(x2) + 1.0)/2.

        elif self.innermost:
            x1 = self.upnorm(self.up(self.down(x)), z)
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x)
            x2 = self.submodule(x1, z)
            x_out = self.upnorm(self.up(x2), z)
            # x_out = self.upnorm(self.model(x), z)
            # x_out = self.pad(x_out, x.size())

            return torch.cat([x_out, x], 1)



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlockShallow(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='replicate'):
        super(UnetBlockShallow, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.pad = padding()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)

        # num_in_groups = inner_nc//16
        # num_out_groups = outer_nc//16

        downnorm = norm_layer() if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer() if norm_layer is not None else None

        if outermost:
            downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=2, padding=p)]

            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Sigmoid()]
            model = down + [submodule] + up

        elif innermost:

            down = [downrelu] #+ downconv
            up = [uprelu]
            if upnorm is not None:
                up += [upnorm]

            innver_conv = [nn.Conv2d(input_nc, outer_nc*2, kernel_size=1, stride=1, padding=0)]
            model = down + innver_conv + up
        else:
            downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=2, padding=p)]

            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):

        if self.outermost or self.innermost:
            # print('self.innermost', self.innermost, x.size())
            return self.model(x)
        else:           
            x_out = self.model(x)
            x_out = self.pad(x_out, x.size())

            return torch.cat([x_out, x], 1)



class UnetBlock_z_middle(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='bilinear', 
                 padding_type='reflect'):
        super(UnetBlock_z_middle, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=3, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()


        num_in_groups = inner_nc//16
        num_out_groups = outer_nc//16

        # downnorm = norm_layer(num_groups=num_in_groups, num_channels=inner_nc) if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(num_out_groups, outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(num_in_groups, inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(num_out_groups, outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z=None):
        # print(x.size())
        # if self.innermost:
            # print('x ', x.size())
            # print('z ', z.size())
            # sys.exit()

            # z_img = z.view(z.size(0), z.size(1), 1, 1).repeat(x.size(0), 1, x.size(2), x.size(3)) #.expand(z.size(0), z.size(1), x.size(2), x.size(3))
            # x_and_z = torch.cat([x, z_img], 1)
        # else:
        print('hello world middle')
        x_and_z = x
        # sys.exit()

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        # print(x.size())
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


def define_E(input_nc, output_nc, ndf, netE,
             norm='instance', nl='lrelu',
             init_type='xavier', init_gain=0.02, vaeLike=False):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    # if netE == 'resnet_128':
        # net = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
                       # nl_layer=nl_layer, vaeLike=vaeLike)
    if netE == 'resnet_256':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    # elif netE == 'conv_128':
        # net = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer,
                        # nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_256':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    # else:
        # raise NotImplementedError('Encoder model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, [])



class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output

def print_network(net_):
    num_params = 0
    for param in net_.parameters():
        num_params += param.numel()
    print(net_)
    print('Total number of parameters: %d' % num_params)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=5,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]

        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer()] #, nn.AvgPool2d(8)]

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(1024, output_nc)])

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # print('x ', x.size())
        x_conv = self.conv(x)
        x_conv = self.avgpool(x_conv)

        conv_flat = x_conv.view(x.size(0), -1)

        # print('conv_flat ', conv_flat.size())
        # sys.exit()

        output = self.fc(conv_flat)

        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output

        return output


# class EqualLR:
#     def __init__(self, name):
#         self.name = name

#     def compute_weight(self, module):
#         weight = getattr(module, self.name + '_orig')
#         fan_in = weight.data.size(1) * weight.data[0][0].numel()

#         return weight * sqrt(2 / fan_in)

#     @staticmethod
#     def apply(module, name):
#         fn = EqualLR(name)

#         weight = getattr(module, name)
#         del module._parameters[name]
#         module.register_parameter(name + '_orig', nn.Parameter(weight.data))
#         module.register_forward_pre_hook(fn)

#         return fn

#     def __call__(self, module, input):
#         weight = self.compute_weight(module)
#         setattr(module, self.name, weight)

# def equal_lr(module, name='weight'):
#     EqualLR.apply(module, name)

#     return module

# class EqualLinear(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()

#         linear = nn.Linear(in_dim, out_dim)
#         linear.weight.data.normal_()
#         linear.bias.data.zero_()

#         self.linear = equal_lr(linear)

#     def forward(self, input):
#         return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.linear = nn.Linear(style_dim, in_channel*2)
        self.linear.weight.data.normal_()
        self.linear.bias.data.zero_()
        # self.style = EqualLinear(style_dim, in_channel * 2)

        self.linear.bias.data[:in_channel] = 1
        self.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.linear(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

# class AdaptiveInstanceNorm(nn.Module):
#     def __init__(self, in_channel, style_dim):
#         super().__init__()
#         mlp_dim = 256
#         self.norm = nn.InstanceNorm2d(in_channel)
#         self.linear_1 = nn.Linear(style_dim, mlp_dim)
#         self.linear_2 = nn.Linear(mlp_dim, in_channel*2)

#         self.linear_1.weight.data.normal_()
#         self.linear_1.bias.data.zero_()
#         self.linear_2.weight.data.normal_()
#         self.linear_2.bias.data.zero_()

#         # self.style = EqualLinear(style_dim, in_channel * 2)
#         self.relu1 = nn.ReLU(inplace=True)

#         # self.linear_2.bias.data[:in_channel] = 1
#         # self.linear_2.bias.data[in_channel:] = 0

#     def forward(self, input, style):
#         style = self.linear_2(self.relu1(self.linear_1(style))).unsqueeze(2).unsqueeze(3)
#         gamma, beta = style.chunk(2, 1)

#         out = self.norm(input)
#         out = gamma * out + beta

#         return out


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1=nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1=nn.ReLU(inplace=True)
        
        self.conv2=nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2=nn.ReLU(inplace=True)
        self.max1=nn.AvgPool2d(kernel_size=2, stride=2)
            
        self.conv3=nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.relu3=nn.ReLU(inplace=True)
            
        self.conv4=nn.Conv2d(128, 128,  kernel_size=3, padding=1, bias=True)
        self.relu4=nn.ReLU(inplace=True)
        self.max2=nn.AvgPool2d(kernel_size=2, stride=2)
            
        self.conv5=nn.Conv2d(128, 256,  kernel_size=3, padding=1, bias=True)
        self.relu5=nn.ReLU(inplace=True)
            
        self.conv6=nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=True)
        self.relu6=nn.ReLU(inplace=True)
            
        self.conv7=nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=True)
        self.relu7=nn.ReLU(inplace=True)
            
        self.conv8=nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=True)
        self.relu8=nn.ReLU(inplace=True)
        self.max3=nn.AvgPool2d(kernel_size=2, stride=2)
            
        self.conv9=nn.Conv2d(256, 512,  kernel_size=3, padding=1, bias=True)
        self.relu9=nn.ReLU(inplace=True)
            
        self.conv10=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu10=nn.ReLU(inplace=True)
            
        self.conv11=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu11=nn.ReLU(inplace=True)
            
        self.conv12=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu12=nn.ReLU(inplace=True)
        self.max4=nn.AvgPool2d(kernel_size=2, stride=2)
            
        self.conv13=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu13=nn.ReLU(inplace=True)
            
        self.conv14=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu14=nn.ReLU(inplace=True)
            
        self.conv15=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu15=nn.ReLU(inplace=True)
            
        self.conv16=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu16=nn.ReLU(inplace=True)
        self.max5=nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, return_style):
        
        out1 = self.conv1(x)
        out2 = self.relu1(out1)

        out3 = self.conv2(out2)
        out4 = self.relu2(out3)
        out5 = self.max1(out4)
        
        out6=self.conv3(out5)
        out7=self.relu3(out6)            
        out8=self.conv4(out7)
        out9=self.relu4(out8)
        out10=self.max2(out9)        
        out11=self.conv5(out10)
        out12=self.relu5(out11)           
        out13=self.conv6(out12)
        out14=self.relu6(out13)            
        out15=self.conv7(out14)
        out16=self.relu7(out15)            
        out17=self.conv8(out16)
        out18=self.relu8(out17)
        out19=self.max3(out18)         
        out20=self.conv9(out19)
        out21=self.relu9(out20)            
        out22=self.conv10(out21)
        out23=self.relu10(out22)            
        out24=self.conv11(out23)
        out25=self.relu11(out24)           
        out26=self.conv12(out25)
        out27=self.relu12(out26)
        out28=self.max4(out27)
        out29=self.conv13(out28)
        out30=self.relu13(out29)
        out31=self.conv14(out30)
        out32=self.relu14(out31)            

        if return_style > 0:
            return [out2, out7, out12, out21, out30]
        else:
            return out4, out9, out14, out23, out32


def vggnet(pretrained=False, model_root=None, **kwargs):
    model = VGG19(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
    return model


class JointLoss(nn.Module):
    def __init__(self, opt):
        super(JointLoss, self).__init__()
        self.opt = opt
        self.num_scales = 4
        self.total_loss = None

        with torch.no_grad():
            self.Net = vggnet(pretrained=False, model_root=None)
            self.Net = self.Net.cuda(1)
            vgg_rawnet = sio.loadmat(self.opt.root + '/feature_mpi/imagenet-vgg-verydeep-19.mat')
            vgg_layers = vgg_rawnet['layers'][0]
            layers=[0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
            att=['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13', 'conv14', 'conv15', 'conv16']
            S = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
            for L in range(16):
                getattr(self.Net, att[L]).weight=nn.Parameter(torch.from_numpy(vgg_layers[layers[L]][0][0][2][0][0]).permute(3,2,0,1).cuda(1))
                getattr(self.Net, att[L]).bias=nn.Parameter(torch.from_numpy(vgg_layers[layers[L]][0][0][2][0][1]).view(S[L]).cuda(1))
            self.Net = self.Net.eval()

    def compute_gradient_loss(self, gt_n_unit, pred_n_unit, mask):
        assert(pred_n_unit.size(1) == gt_n_unit.size(1))

        n_diff = pred_n_unit - gt_n_unit
        mask_rep = mask.repeat(1, gt_n_unit.size(1), 1, 1)

        assert(pred_n_unit.size(1) == mask_rep.size(1))

        # vertical gradient
        v_gradient = torch.abs(n_diff[:, :, :-2,:] - n_diff[:, :, 2:,:])
        v_mask = torch.mul(mask_rep[:, :, :-2,:], mask_rep[:, :, 2:,:])
        v_gradient = torch.mul(v_gradient, v_mask)
        # horizontal gradient
        h_gradient = torch.abs(n_diff[:, :, :, :-2] - n_diff[:, :, :, 2:])
        h_mask = torch.mul(mask_rep[:, :, :, :-2], mask_rep[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        N = torch.sum(h_mask) + torch.sum(v_mask) + EPSILON
        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss/(N)
        return gradient_loss

    def compute_mae_loss(self, pred_img, gt_img, gt_mask):
        gt_mask_rep = gt_mask.repeat(1, pred_img.size(1), 1, 1)

        assert(pred_img.size(1) == gt_mask_rep.size(1))
        assert(pred_img.size(1) == gt_img.size(1))

        num_valid_pixels = torch.sum(gt_mask_rep) + EPSILON

        loss = torch.sum( torch.abs(pred_img - gt_img) * gt_mask_rep)/num_valid_pixels
        return loss

    def compute_psnr(self, pred_img, gt_img, gt_mask):

        gt_mask_rep = gt_mask.repeat(1, pred_img.size(1), 1, 1)
        num_valid_pixels = torch.sum(gt_mask_rep) + EPSILON

        mse = torch.sum( torch.pow(pred_img - gt_img, 2) * gt_mask_rep)/num_valid_pixels

        psnr = 10 * torch.log10(1./mse)
        
        return psnr

    def compute_lpips(self, pred_img, gt_img, gt_mask, lpips_model):

        gt_mask_rep = gt_mask.repeat(1, pred_img.size(1), 1, 1)
        # num_valid_pixels = torch.sum(gt_mask_rep) + EPSILON
        pred_img = pred_img * gt_mask_rep
        gt_img = gt_img * gt_mask_rep

        lpips = lpips_model.forward(pred_img, gt_img, gt_mask, normalize=True).item()

        return lpips


    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / (SSIM_d)

        return torch.clamp((SSIM), 0, 1)


    def compute_ssim(self, pred_img, gt_img, gt_mask):

        gt_mask_rep = gt_mask.repeat(1, pred_img.size(1), 1, 1)

        gt_mask_rep_crop = gt_mask_rep[:, :, 1:-1, 1:-1]

        # print('self.SSIM(pred_img, gt_img) ', self.SSIM(pred_img, gt_img).size())
        # sys.exit()

        # ssim term
        ssim = torch.sum(gt_mask_rep_crop * self.SSIM(pred_img, gt_img))/(torch.sum(gt_mask_rep_crop) + EPSILON) 

        return ssim


    def scale_invariance_mse(self, pred_log_d, gt_log_d, mask):
        assert(pred_log_d.size(1) == gt_log_d.size(1))
        assert(mask.size(1) == pred_log_d.size(1))

        final_mask = mask #* (pred_d.detach() > EPSILON).float().cuda()

        # pred_log_d = torch.log(torch.clamp(pred_d, min=EPSILON))
        # gt_log_d = torch.log(torch.clamp(gt_d, min=EPSILON))

        num_valid_pixels = torch.sum(final_mask) + EPSILON
        log_d_diff = pred_log_d - gt_log_d
        log_d_diff = log_d_diff * final_mask
        s1 = torch.sum( torch.pow(log_d_diff, 2) )/num_valid_pixels 
        s2 = (torch.sum(log_d_diff) * torch.sum(log_d_diff) )/(num_valid_pixels*num_valid_pixels)  

        data_loss = s1 - s2
        
        return data_loss

    def compute_style_loss(self, pred_img, real_img):
        def compute_gram_matrix(input):
            a, b, c, d = input.size()  # a=batch size(=1)
            features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
            G = torch.mm(features, features.t())  # compute the gram product

            # we 'normalize' the values of the gram matrix
            # by dividing by the number of element in each feature maps.
            return G.div(a* b * c * d)

        aa = np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        bb = Variable(torch.from_numpy(aa).float().permute(0,3,1,2).cuda(1))

        real_img_sb = real_img * 255. - bb
        pred_img_sb = pred_img * 255. - bb

        real_style = self.Net(real_img_sb, return_style=1)
        fake_style = self.Net(pred_img_sb, return_style=1)
        style_loss = 0.

        for x, y in zip(fake_style, real_style):
            gram_fake = compute_gram_matrix(x)
            gram_real = compute_gram_matrix(y)
            style_loss += torch.mean(torch.abs(gram_real - gram_fake))

        return style_loss/(2048.)

    def compute_Perceptual_Loss(self, pred_img, real_img, mask):
        def compute_error(real, fake):
            E = torch.mean(torch.abs(real - fake) )
            return E

        aa = np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        bb = Variable(torch.from_numpy(aa).float().permute(0,3,1,2).cuda(1))

        real_img_sb = real_img * mask * 255. - bb
        pred_img_sb = pred_img * mask * 255. - bb

        out3_r, out8_r, out13_r, out22_r, out33_r = self.Net(real_img_sb, return_style=0)
        out3_f, out8_f, out13_f, out22_f, out33_f = self.Net(pred_img_sb, return_style=0)

        E0 = compute_error(real_img_sb, pred_img_sb)
        # mask_3 = nn.functional.interpolate(mask, size=[out3_r.size(2), out3_r.size(3)], mode='nearest')
        E1 = compute_error(out3_r, out3_f)/2.6
        # mask_8 = nn.functional.interpolate(mask, size=[out8_r.size(2), out8_r.size(3)], mode='nearest')
        E2 = compute_error(out8_r, out8_f)/4.8
        # mask_13 = nn.functional.interpolate(mask, size=[out13_r.size(2), out13_r.size(3)], mode='nearest')
        E3 = compute_error(out13_r, out13_f)/3.7
        # mask_22 = nn.functional.interpolate(mask, size=[out22_r.size(2), out22_r.size(3)], mode='nearest')
        E4 = compute_error(out22_r, out22_f)/5.6
        # mask_33 = nn.functional.interpolate(mask, size=[out33_r.size(2), out33_r.size(3)], mode='nearest')
        E5 = compute_error(out33_r, out33_f)*10/ 1.5

        total_loss = (E0+E1+E2+E3+E4+E5)/255.

        return total_loss


    def __call__(self, target_imgs, pred_rgb, latent_z, targets, n_iter):

        gt_mask = Variable(targets['gt_mask'].cuda(), requires_grad=False).unsqueeze(1) #* visible_mask
    
        img_loss = self.compute_mae_loss(pred_rgb, target_imgs, gt_mask)

        latent_reg_loss = self.opt.w_z_reg * torch.mean(latent_z**2)

        print('img_loss ', img_loss.item(), ' latent_reg_loss ', latent_reg_loss.item())

        self.total_loss = img_loss + latent_reg_loss

        return {'total_loss': self.total_loss.item(),
                'img_loss':img_loss.item(),
                'latent_reg_loss':latent_reg_loss.item()}

    def get_loss_var(self):
        return self.total_loss


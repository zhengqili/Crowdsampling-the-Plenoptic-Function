from __future__ import division 
import numpy as np
import torch
import os
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks
import sys, traceback
import os.path
import cv2
import torch.nn as nn
import math

import models
import models.encoder as models_spade
import models.projector as projector


from torch.nn import init

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPSILON = 1e-8

class RenderModel(BaseModel):
    def name(self):
        return 'RenderModel'

    def __init__(self, opt, _isTrain):
        BaseModel.initialize(self, opt)
        self.num_scales = 4

        pretrain_dir = opt.local_dir + '/pretrain_alpha_new/'

        if opt.dataset == 'trevi':
            pretrain_mpi_path = pretrain_dir + 'latest_trevi_feature_mpi_exp_clean_model_resnet_encoder_lr_0.001_num_mpi_planes_64_max_depth_4_min_depth_1_fov_70_stage_2_use_log_l1_0ref_feature_mpi.npy'
        elif opt.dataset == 'pantheon':
            pretrain_mpi_path = pretrain_dir + 'latest_pantheon_feature_mpi_exp_clean_model_resnet_encoder_lr_0.001_num_mpi_planes_64_max_depth_25_min_depth_1_fov_65_stage_2_use_log_l1_0ref_feature_mpi.npy'
        elif opt.dataset == 'coeur':
            pretrain_mpi_path = pretrain_dir + 'latest_coeur_feature_mpi_exp_clean_model_resnet_encoder_lr_0.001_num_mpi_planes_64_max_depth_20_min_depth_1_fov_65_stage_2_use_log_l1_0ref_feature_mpi.npy'
        elif opt.dataset == 'rock':
            pretrain_mpi_path = pretrain_dir + 'latest_rock_feature_mpi_exp_clean_model_resnet_encoder_lr_0.001_num_mpi_planes_64_max_depth_75_min_depth_1_fov_70_stage_2_use_log_l1_0ref_feature_mpi.npy'
        elif opt.dataset == 'navona':
            pretrain_mpi_path = pretrain_dir + 'latest_navona_feature_mpi_exp_clean_model_resnet_encoder_lr_0.001_num_mpi_planes_64_max_depth_25_min_depth_1_fov_70_stage_2_use_log_l1_0ref_feature_mpi.npy'

        self.criterion_joint = networks.JointLoss(opt) 

        ref_albedo_mpi = np.load(pretrain_mpi_path)
        self.ref_albedo_mpi = Variable(torch.from_numpy(np.ascontiguousarray(ref_albedo_mpi)).contiguous().float().cuda(), requires_grad=False)
        self.ref_albedo_mpi = torch.sigmoid(self.ref_albedo_mpi)
        self.ref_albedo_mpi[0, -1:, :, :] = 1.0

        mpi_planes = self.generate_mpi_depth_planes(opt.min_depth, opt.max_depth, opt.num_mpi_planes)
        self.mpi_planes = Variable(torch.from_numpy(np.ascontiguousarray(mpi_planes)).contiguous().float().cuda(), requires_grad=False)

        self.K_ref = self.create_ref_intrinsic(opt.mpi_w, opt.mpi_h, opt.ref_fov)
        self.ref_feature_img = self.create_feature_mpi(opt.num_mpi_planes, opt.num_mpi_f, opt.mpi_w, opt.mpi_h)
        self.opt = opt

        self.ref_albedo_rgba_mpi_small = torch.nn.functional.interpolate(self.ref_albedo_mpi, [512, 512], mode='bilinear').to(1)
        self.composite_albedo_ref = projector.over_composite(self.ref_albedo_rgba_mpi_small[:, :-1, : , :], self.ref_albedo_rgba_mpi_small[:, -1:, :, :], premultiply_alpha=0).unsqueeze(0)
        
        self.ref_feature_img_small = torch.nn.functional.interpolate(self.ref_feature_img, [512, 512], mode='bilinear').to(1)

        appearace_encoder = models_spade.AppearanceEncoder(opt)

        if opt.where_add == 'adain':
            num_input_features = 3 + opt.num_mpi_f + 1
            neural_render = networks.define_G(input_nc=num_input_features, output_nc=3, nz=0, ngf=opt.ngf,
                                              norm=opt.norm_G, nl=opt.nl, use_dropout=False, init_type=opt.init_type, 
                                              init_gain=opt.init_gain, where_add=opt.where_add, 
                                              upsample=opt.upsample, style_dim=opt.num_latent_f)
        else:
            num_input_features = 3 + opt.num_mpi_f + opt.num_latent_f + 1
            neural_render = networks.define_G(input_nc=num_input_features, output_nc=3, nz=0, ngf=opt.ngf,
                                              norm=opt.norm_G, nl=opt.nl, use_dropout=False, init_type=opt.init_type, 
                                              init_gain=opt.init_gain, where_add=opt.where_add, upsample=opt.upsample)

 
        if opt.dataset == 'trevi':
            model_name = '_best_trevi_feature_mpi_exp_independent_warp_model_munit_encoder_lr_0.0004_use_gan_1_use_vgg_loss_1_warp_src_img_1_where_add_adain'
        elif opt.dataset == 'pantheon':
            model_name ='_best_pantheon_feature_mpi_exp_independent_warp_model_munit_encoder_lr_0.0004_use_gan_1_use_vgg_loss_1_warp_src_img_1_where_add_adain'
        elif opt.dataset == 'coeur':
            model_name = '_best_coeur_feature_mpi_exp_independent_warp_model_munit_encoder_lr_0.0004_use_gan_1_use_vgg_loss_1_warp_src_img_1_where_add_adain'
        elif opt.dataset == 'rock':
            model_name = '_best_rock_feature_mpi_exp_independent_warp_model_munit_encoder_lr_0.0004_use_gan_1_use_vgg_loss_1_warp_src_img_1_where_add_adain'
        elif opt.dataset == 'navona':
            model_name = '_best_navona_feature_mpi_exp_independent_warp_model_munit_encoder_lr_0.0004_use_gan_1_use_vgg_loss_1_warp_src_img_1_where_add_adain'

        feature_mpi_name = opt.local_dir + '/deep_mpi/' + model_name[1:] + 'ref_feature_img.npy'

        appearace_encoder.load_state_dict(self.load_network(appearace_encoder, 'E', model_name))
        neural_render.load_state_dict(self.load_network(neural_render, 'G', model_name))

        self.ref_feature_img = Variable(torch.from_numpy(np.ascontiguousarray(np.load(feature_mpi_name))).contiguous().float().cuda(), requires_grad=False)

        self.netE = torch.nn.parallel.DataParallel(appearace_encoder.cuda(1), device_ids = [1])
        self.netG = torch.nn.parallel.DataParallel(neural_render.cuda(1), device_ids = [1, 2, 3])

        self.netE.eval()
        self.netG.eval()

        print('---------- Encoder Networks initialized -------------')
        networks.print_network(self.netE)

        print('---------- neural_render Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')


    def create_ref_intrinsic(self, mpi_w, mpi_h, fov):
        f = float(mpi_w)/(2. * np.tan(np.radians(fov)/2.))        
        intrinsic = np.eye(3)
        intrinsic[0, 0] = intrinsic[1, 1] = f
        intrinsic[0, 2] = (mpi_w)/2.
        intrinsic[1, 2] = (mpi_h)/2.
        intrinsic = Variable(torch.from_numpy(np.ascontiguousarray(intrinsic)).contiguous().float().cuda().unsqueeze(0), requires_grad=False)

        return intrinsic

    def create_feature_mpi(self, num_planes, num_features, mpi_w, mpi_h):
        features_mpi = np.zeros((num_planes, num_features, mpi_h, mpi_w))
        mean_features_mpi = nn.Parameter((torch.Tensor(features_mpi).float().cuda())) #, requires_grad=True)
        init.normal_(mean_features_mpi.data, 0.0, 0.02)

        return mean_features_mpi


    def set_input(self, targets):
        self.targets = targets

    def evaluate_all(self, lpips_model):

        with torch.no_grad():

            K_a = Variable(self.targets['K_a'].cuda(), requires_grad=False)
            R_ar = Variable(self.targets['R_ar'].cuda(), requires_grad=False)
            t_ar = Variable(self.targets['t_ar'].cuda(), requires_grad=False)
    
            img_a_full = Variable(self.targets['img_a_full'].cuda(1), requires_grad=False)            
            warp_img_a = Variable(self.targets['warp_img_a'].cuda(1), requires_grad=False)
            img_a = Variable(self.targets['img_a'].cuda(1), requires_grad=False)

            pred_render_rgb_a, pred_albedo_a, \
            warp_mpi_depth_a, proj_ref_mask_a = self.render_mpi_imgs_func(img_a_full, warp_img_a, img_a,
                                                                          K_a, R_ar, t_ar)   


            gt_mask_a = Variable(self.targets['gt_mask_a'].cuda(1), requires_grad=False).unsqueeze(1)
            gt_ref_mask_a = proj_ref_mask_a * gt_mask_a

            if torch.sum(gt_ref_mask_a) < 10:
                return 0,0,0,0

            # compute L1 error 
            l1_error = self.criterion_joint.compute_mae_loss(pred_render_rgb_a, img_a, gt_ref_mask_a)
            # compute PSNR 
            psnr = self.criterion_joint.compute_psnr(pred_render_rgb_a, img_a, gt_ref_mask_a)
            # compute lpips
            lpips = self.criterion_joint.compute_lpips(pred_render_rgb_a, img_a, gt_ref_mask_a, lpips_model)

            return l1_error, psnr, lpips, 1


    def get_latent_feature(self, img_a_full, warp_img_a):
        composite_feature_ref = projector.over_composite(self.ref_feature_img_small.detach(), self.ref_albedo_rgba_mpi_small[:, -1:, :, :], premultiply_alpha=0).unsqueeze(0)

        warp_concat = torch.cat([warp_img_a, self.composite_albedo_ref, composite_feature_ref], dim=1)

        latent_z_a = self.netE.forward(warp_concat, img_a_full)

        return latent_z_a

    def infer_app_mpi_from_mpi(self, ref_imgs, img_a, 
                               latent_z_a, 
                               K_ref, K_a,
                               R_ar, t_ar, depths):

        proj_ref_imgs_a = self.mpi_render_view(ref_imgs, img_a, R_ar, t_ar, K_ref, K_a, depths).to(1)
        # proj_ref_imgs_b = self.mpi_render_view(ref_imgs, img_b, R_br, t_br, K_ref, K_b, depths).to(1)

        proj_ref_features_a = proj_ref_imgs_a[:, 0:self.opt.num_mpi_f, : ,:]
        proj_ref_albedo_a = proj_ref_imgs_a[:, self.opt.num_mpi_f:-1, : ,:]
        proj_ref_alphas_a = proj_ref_imgs_a[:, -1:, :, :]

        composite_albedo_a = projector.over_composite(proj_ref_albedo_a, proj_ref_alphas_a, premultiply_alpha=0).unsqueeze(0)

        depth_planes = depths.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        proj_depth_planes_a = depth_planes.repeat(1, 1, proj_ref_imgs_a.size(2), proj_ref_imgs_a.size(3))
        warp_mpi_depth_a = self.create_mpi_depth(proj_depth_planes_a.to(1), proj_ref_alphas_a)

        proj_ref_mask_a = torch.sum(proj_ref_alphas_a > 1e-8, dim=0, keepdim=True)
        proj_ref_mask_a = (proj_ref_mask_a > self.opt.num_mpi_planes*3//4).type(torch.cuda.FloatTensor)

        # render image at viewpoint a, condition on image a
        latent_z_a_rep = latent_z_a.view(latent_z_a.size(0), latent_z_a.size(1), 1, 1).repeat(proj_ref_imgs_a.size(0), 1, proj_ref_imgs_a.size(2), proj_ref_imgs_a.size(3)) #.expand(z.size(0), z.size(1), x.size(2), x.size(3))        
        
        buffer_a = torch.cat([proj_ref_features_a, proj_ref_albedo_a, proj_ref_alphas_a], dim=1)
        proj_ref_inputs_concat_a = torch.cat([buffer_a, latent_z_a_rep], dim=1)

        render_rgb_mpi_a = self.netG.forward(proj_ref_inputs_concat_a)

        composite_rgb_img_a = projector.over_composite(render_rgb_mpi_a, proj_ref_alphas_a, premultiply_alpha=0).unsqueeze(0)

        return composite_rgb_img_a, torch.cat([render_rgb_mpi_a, proj_ref_alphas_a], dim=1)
            

    def interpolate_appearance(self, save_root_dir):

        def get_latent_feature(img_a_full, warp_img_a):
            composite_feature_ref = projector.over_composite(self.ref_feature_img_small.detach(), self.ref_albedo_rgba_mpi_small[:, -1:, :, :], premultiply_alpha=0).unsqueeze(0)
            warp_concat = torch.cat([warp_img_a, self.composite_albedo_ref, composite_feature_ref], dim=1)

            latent_z_a = self.netE.forward(warp_concat, img_a_full)

            return latent_z_a


        num_interpolations = 20

        # img_path = self.targets['img_path'][0]
        save_dir = save_root_dir + self.targets['img_path_a'][0] + '*' + self.targets['img_path_b'][0] + '*' + self.targets['img_path_c'][0] + '/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with torch.no_grad():

            K_c = Variable(self.targets['K_c'].cuda(), requires_grad=False)
            R_cr = Variable(self.targets['R_cr'].cuda(), requires_grad=False)
            t_cr = Variable(self.targets['t_cr'].cuda(), requires_grad=False)

            img_a_full = Variable(self.targets['img_a_full'].cuda(1), requires_grad=False)
            img_b_full = Variable(self.targets['img_b_full'].cuda(1), requires_grad=False)

            img_a = Variable(self.targets['img_a'], requires_grad=False)
            img_b = Variable(self.targets['img_b'], requires_grad=False)
            img_c = Variable(self.targets['img_c'], requires_grad=False)

            warp_img_a = Variable(self.targets['warp_img_a'].cuda(1), requires_grad=False)
            warp_img_b = Variable(self.targets['warp_img_b'].cuda(1), requires_grad=False)

            img_a_np = img_a[0].data.cpu().numpy().transpose(1, 2, 0)
            img_b_np = img_b[0].data.cpu().numpy().transpose(1, 2, 0)
            img_c_np = img_c[0].data.cpu().numpy().transpose(1, 2, 0)

            cv2.imwrite(save_dir + '/%s'%self.targets['img_path_a'][0], np.uint8(img_a_np[:, :, ::-1] * 255) )
            cv2.imwrite(save_dir + '/%s'%self.targets['img_path_b'][0], np.uint8(img_b_np[:, :, ::-1] * 255) )
            cv2.imwrite(save_dir + '/%s'%self.targets['img_path_c'][0], np.uint8(img_c_np[:, :, ::-1] * 255) )

            ref_mpi_concat = torch.cat([self.ref_feature_img, self.ref_albedo_mpi], dim=1)

            latent_z_a = get_latent_feature(img_a_full, warp_img_a)
            latent_z_b = get_latent_feature(img_b_full, warp_img_b)

            delta_f = (latent_z_b - latent_z_a) / num_interpolations
            for delta_iter in range(num_interpolations + 1):
                print(delta_iter)
                x_app_embedding = latent_z_a + delta_iter * delta_f

                pred_render_rgb_a, warp_alphas_a, warp_mpi_depth_a, proj_ref_mask_a,  = self.infer_rgb_from_mpi(ref_mpi_concat, img_c,
                                                                                                                x_app_embedding,
                                                                                                                self.K_ref, K_c, 
                                                                                                                R_cr, t_cr, self.mpi_planes)
                pred_render_rgb_a = pred_render_rgb_a * proj_ref_mask_a
                pred_render_rgb_np = pred_render_rgb_a[0].data.cpu().numpy().transpose(1, 2, 0)
                cv2.imwrite(save_dir + 'ours_inteporlation_%d.jpg'%delta_iter, np.uint8(pred_render_rgb_np[:, :, ::-1] * 255) )



    # def viz_render_rgb(self, save_root_dir):
    #     with torch.no_grad():
    #         img_name = self.targets['img_path_a'][0]
    #         save_dir = save_root_dir + img_name + '/'
            
    #         if not os.path.exists(save_dir):
    #              os.makedirs(save_dir)

    #         gt_mask_a = Variable(self.targets['gt_mask_a'].cuda(), requires_grad=False).unsqueeze(1)

    #         R_ar = Variable(self.targets['R_ar'].cuda(), requires_grad=False)
    #         t_ar = Variable(self.targets['t_ar'].cuda(), requires_grad=False)

    #         K_a = Variable(self.targets['K_a'].cuda(), requires_grad=False)
    #         # K_b = Variable(self.targets['K_b'].cuda(), requires_grad=False)

    #         R_ar = Variable(self.targets['R_ar'].cuda(), requires_grad=False)
    #         t_ar = Variable(self.targets['t_ar'].cuda(), requires_grad=False)

    #         img_a_full = Variable(self.targets['img_a_full'].cuda(1), requires_grad=False)
    #         warp_img_a = Variable(self.targets['warp_img_a'].cuda(1), requires_grad=False)
    #         img_a = Variable(self.targets['img_a'].cuda(1), requires_grad=False)

    #         pred_render_rgb_a, pred_albedo_a, \
    #         warp_mpi_depth_a, proj_ref_mask_a = self.render_mpi_imgs_func(img_a_full, warp_img_a, img_a, K_a, R_ar, t_ar)

    #         pred_render_rgb_a = pred_render_rgb_a * proj_ref_mask_a #* gt_mask_a

    #         pred_render_rgb_a_np = pred_render_rgb_a.data[0].cpu().numpy().transpose(1, 2, 0)
    #         img_a_np = img_a[0].data.cpu().numpy().transpose(1, 2, 0)

    #         cv2.imwrite(save_dir + '/gt_rgb.jpg', np.uint8(img_a_np[:, :, ::-1] * 255) )
    #         cv2.imwrite(save_dir + '/render_rgb.jpg', np.uint8(pred_render_rgb_a_np[:, :, ::-1] * 255) )


    # def viz_mpi_inference(self, save_root_dir):
    #     with torch.no_grad():
    #         img_name = self.targets['img_name'][0]
    #         save_dir = save_root_dir + img_name + '/'
    #         if not os.path.exists(save_dir):
    #              os.makedirs(save_dir)

    #         gt_mask_a = Variable(self.targets['gt_mask_a'].cuda(), requires_grad=False).unsqueeze(1)

    #         K_a = Variable(self.targets['K_a'].cuda(), requires_grad=False)
    #         K_b = Variable(self.targets['K_b'].cuda(), requires_grad=False)

    #         R_ar = Variable(self.targets['R_ar'].cuda(), requires_grad=False)
    #         t_ar = Variable(self.targets['t_ar'].cuda(), requires_grad=False)

    #         # R_br = Variable(self.targets['R_br'].cuda(), requires_grad=False)
    #         # t_br = Variable(self.targets['t_br'].cuda(), requires_grad=False)

    #         img_a_full = Variable(self.targets['img_a_full'].cuda(), requires_grad=False)
    #         img_b_full = Variable(self.targets['img_b_full'].cuda(), requires_grad=False)

    #         img_a_np = img_a_full[0].data.cpu().numpy().transpose(1, 2, 0)
    #         img_b_np = img_b_full[0].data.cpu().numpy().transpose(1, 2, 0)

    #         cv2.imwrite(save_dir + '/img_a.png', np.uint8(img_a_np[:, :, ::-1] * 255) )
    #         cv2.imwrite(save_dir + '/img_b.png', np.uint8(img_b_np[:, :, ::-1] * 255) )

    #         latent_z_b = self.netE.forward(img_b_full)
    #         ref_mpi_concat = torch.cat([self.ref_feature_img, self.ref_albedo_mpi], dim=1)

    #         pred_render_rgb_ba, render_rgb_mpi_ba, warp_alphas_a, warp_mpi_depth_ba, proj_ref_mask_a = self.infer_rgb_from_mpi(ref_mpi_concat, img_a_full,
    #                                                                                                                                   latent_z_b,
    #                                                                                                                                   self.K_ref, K_a, 
    #                                                                                                                                   R_ar, t_ar, self.mpi_planes)

    #         render_rgb_mpi_ba = render_rgb_mpi_ba * proj_ref_mask_a * gt_mask_a

    #         render_rgb_mpi = render_rgb_mpi_ba.data.cpu().numpy()
    #         warp_alphas = warp_alphas_a.data.cpu().numpy()

    #         warp_mpi_depth_np = warp_mpi_depth_ba.data.cpu().numpy()
    #         warp_mpi_depth_np = warp_mpi_depth_np[0]
    #         plt.imshow(warp_mpi_depth_np)
    #         plt.axis('off')
    #         plt.savefig(save_dir + 'depth.png')
    #         plt.close()

    #         for i in range(self.opt.num_mpi_planes):
    #             print('write plane ', img_name, i)
    #             render_rgb_mpi_i = render_rgb_mpi[i].transpose(1, 2, 0)
    #             warp_alphas_i = warp_alphas[i, 0]

    #             print('render_rgb_mpi_i ', render_rgb_mpi_i.shape)
    #             print('warp_alphas_i ', warp_alphas_i.shape)

    #             render_rgba_mpi_i = cv2.cvtColor(render_rgb_mpi_i, cv2.COLOR_RGB2RGBA)
    #             render_rgba_mpi_i[:, :, 3] = warp_alphas_i
    #             render_rgba_mpi_i = cv2.cvtColor(render_rgba_mpi_i, cv2.COLOR_RGBA2BGRA)

    #             cv2.imwrite(save_dir + 'rgba_%02d.png'%i, np.uint8(render_rgba_mpi_i * 255))

    def render_wander(self, save_root_dir):

        num_frames = 90
        img_path = self.targets['img_path_c'][0]
        img_name = img_path.split('/')[-1][:-4]

        save_dir = save_root_dir + self.targets['img_path_a'][0] + '_' + self.targets['img_path_b'][0] + '/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        max_disp = 32.

        max_trans = max_disp / self.targets['K_a'][0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
        output_poses = []

        for i in range(num_frames):
            x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
            y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /2.0 #* 3.0 / 4.0
            z_trans = -max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames)) / 2.0

            i_pose = np.concatenate([
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                    # [np.eye(3), np.array([x_trans, 0., 0.])[:, np.newaxis]], axis=1),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
            ],axis=0)[np.newaxis, :, :]

            output_poses.append(i_pose)

        with torch.no_grad():

            K_a = Variable(self.targets['K_a'].cuda(), requires_grad=False)

            R_ar = Variable(self.targets['R_ar'].cuda(), requires_grad=False)
            t_ar = Variable(self.targets['t_ar'].cuda(), requires_grad=False)

            K_b = Variable(self.targets['K_b'].cuda(), requires_grad=False)

            R_br = Variable(self.targets['R_br'].cuda(), requires_grad=False)
            t_br = Variable(self.targets['t_br'].cuda(), requires_grad=False)
            
            img_b_full = Variable(self.targets['img_b_full'].cuda(), requires_grad=False)
            warp_img_b = Variable(self.targets['warp_img_b'].cuda(), requires_grad=False)

            img_a = Variable(self.targets['img_a'].cuda(), requires_grad=False)
            img_b = Variable(self.targets['img_b'].cuda(), requires_grad=False)

            img_a_np = img_a[0].data.cpu().numpy().transpose(1, 2, 0)
            img_b_np = img_b[0].data.cpu().numpy().transpose(1, 2, 0)

            cv2.imwrite(save_dir + '/%s'%self.targets['img_path_a'][0], np.uint8(img_a_np[:, :, ::-1] * 255) )
            cv2.imwrite(save_dir + '/%s'%self.targets['img_path_b'][0], np.uint8(img_b_np[:, :, ::-1] * 255) )

            latent_z_b = self.get_latent_feature(img_b_full, warp_img_b)
            ref_mpi_concat = torch.cat([self.ref_feature_img, self.ref_albedo_mpi], dim=1)

            w, h = 512, 384

            K_a[:, 0, 2] = K_a[:, 0, 2] * float(w)/img_a.size(3)
            K_a[:, 1, 2] = K_a[:, 1, 2] * float(h)/img_a.size(2)

            scale_factor = 1.25
            K_a_large = K_a.clone()
            K_a_large[:, 0, 2] = K_a[:, 0, 2] * scale_factor
            K_a_large[:, 1, 2] = K_a[:, 1, 2] * scale_factor

            img_a = nn.functional.interpolate(img_a, size=[h, w])
            
            _, render_rgba_mpi_ba = self.infer_app_mpi_from_mpi(ref_mpi_concat, nn.functional.interpolate(img_a, scale_factor=scale_factor),
                                                                latent_z_b.to(1),
                                                                self.K_ref, K_a_large, 
                                                                R_ar, t_ar, self.mpi_planes)

            render_rgba_mpi_ba_2 = render_rgba_mpi_ba.to(2)
            K_a_large_2 = K_a_large.to(2)
            K_a_2 = K_a.to(2)
            mpi_planes_2 = self.mpi_planes.to(2)
            
            for i in range(num_frames):
                print('render ', i)
                pose = output_poses[i]
                R_ta = Variable(torch.from_numpy(np.ascontiguousarray(pose[:, :3, :3])).contiguous().float().cuda(2), requires_grad=False)
                t_ta = Variable(torch.from_numpy(np.ascontiguousarray(pose[:, :3, 3:])).contiguous().float().cuda(2), requires_grad=False)

                proj_ref_imgs_a = self.mpi_render_view(render_rgba_mpi_ba.to(2), img_a, R_ta, t_ta, K_a_large.to(2), K_a.to(2), self.mpi_planes.to(2), cuda_id=2)#.to(1)

                proj_ref_rgb_a = proj_ref_imgs_a[:, :-1, :, :]
                proj_ref_alphas_a = proj_ref_imgs_a[:, -1:, :, :]
                proj_ref_mask_a = torch.sum(proj_ref_alphas_a > 1e-8, dim=0, keepdim=True)
                proj_ref_mask_a = (proj_ref_mask_a > self.opt.num_mpi_planes*3//4).type(torch.cuda.FloatTensor)

                pred_render_rgb_ba = projector.over_composite(proj_ref_rgb_a, proj_ref_alphas_a)

                pred_render_rgb_ba = pred_render_rgb_ba * proj_ref_mask_a
                pred_render_rgb_np = pred_render_rgb_ba[0].data.cpu().numpy().transpose(1, 2, 0) 
                cv2.imwrite(save_dir + '/wander_rgb_ours_%d.png'%i, np.uint8(pred_render_rgb_np[:, :, ::-1] * 255) )
                cv2.imwrite(save_dir + '/wander_rgb_ours_%d.png'%(num_frames+i), np.uint8(pred_render_rgb_np[:, :, ::-1] * 255) )


    def generate_mpi_depth_planes(self, start_depth, end_depth, num_depths):
        """Sample reversed, sorted inverse depths between a near and far plane.
        Args:
          start_depth: The first depth (i.e. near plane distance).
          end_depth: The last depth (i.e. far plane distance).
          num_depths: The total number of depths to create. start_depth and
              end_depth are always included and other depths are sampled
              between them uniformly according to inverse depth.
        Returns:
          The depths sorted in descending order (so furthest first). This order is
          useful for back to front compositing.
        """
        inv_start_depth = 1.0 / start_depth
        inv_end_depth = 1.0 / end_depth
        depths = [start_depth, end_depth]
        for i in range(1, num_depths - 1):
          fraction = float(i) / float(num_depths - 1)
          inv_depth = inv_start_depth + (inv_end_depth - inv_start_depth) * fraction
          depths.append(1.0 / inv_depth)
        depths = sorted(depths)
        return np.array(depths[::-1])

    def mpi_render_view(self, ref_imgs, tgt_imgs, R_tr, t_tr, K_ref, K_src, depths, cuda_id=0):
        """Render a target view from an MPI representation in ref view.
        Args:
          ref_imgs: input feature MPI [#planes, #features, height, width]
          R_tr: rotation from ref to target 
          t_tr: translation from ref to target 
          depths: array of depth for each plane
          K_ref: camera intrinsics [batch, 3, 3]
        Returns:
          rendered view [batch, height, width, 3]
        """
        num_planes = ref_imgs.size(0)

        proj_ref_imgs = projector.projective_forward_homography(ref_imgs, K_src,
                                                                tgt_imgs, K_ref,
                                                                R_tr, t_tr, depths, cuda_id)

        # over after feeding feature to the network
        # composite_imgs = projector.over_composite(proj_ref_imgs)

        return proj_ref_imgs#.unsqueeze(0)

    def create_mpi_depth(self, depth_planes, alphas):
        final_depth = projector.over_composite(depth_planes, alphas)    

        return final_depth

    def viz_new_views(self, ref_imgs, tgt_imgs, latent_z, K_ref, K_tgt, R_tr, t_tr, depths):

        proj_ref_imgs = self.mpi_render_view(ref_imgs, tgt_imgs, R_tr, t_tr, K_ref, K_tgt, depths).to(1)

        # proj_ref_features = proj_ref_imgs[:, :self.opt.num_mpi_f, :, :]
        proj_ref_rgb = proj_ref_imgs[:, self.opt.num_mpi_f:-1, : ,:]
        proj_ref_alphas = proj_ref_imgs[:, -1:, :, :]

        # composite_feature_img = projector.over_composite(proj_ref_features, proj_ref_alphas).unsqueeze(0).repeat(self.opt.num_mpi_planes, 1, 1, 1)

        latent_z_rep = latent_z.view(latent_z.size(0), latent_z.size(1), 1, 1).repeat(proj_ref_imgs.size(0), 1, proj_ref_imgs.size(2), proj_ref_imgs.size(3)) 
    
        depth_planes = depths.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        proj_depth_planes = depth_planes.repeat(1, 1, proj_ref_imgs.size(2), proj_ref_imgs.size(3))
        warp_mpi_depth = self.create_mpi_depth(proj_depth_planes.to(1), proj_ref_alphas)

        # target_pts_3d = self.create_3d_pts(warp_mpi_depth, K_tgt).unsqueeze(0).repeat(self.opt.num_mpi_planes, 1, 1, 1)

        proj_ref_inputs_concat = torch.cat([proj_ref_imgs, latent_z_rep], dim=1)

        render_rgb_mpi = self.netG.forward(proj_ref_inputs_concat)

        composite_rgb_img = projector.over_composite(render_rgb_mpi, proj_ref_alphas)
        composite_rgb_img = composite_rgb_img.unsqueeze(0)

        return composite_rgb_img

    def infer_rgb_from_mpi(self, ref_imgs, img_a, 
                           latent_z_a, 
                           K_ref, K_a,
                           R_ar, t_ar, depths):

        proj_ref_imgs_a = self.mpi_render_view(ref_imgs, img_a, R_ar, t_ar, K_ref, K_a, depths).to(1)
        # proj_ref_imgs_b = self.mpi_render_view(ref_imgs, img_b, R_br, t_br, K_ref, K_b, depths).to(1)

        proj_ref_features_a = proj_ref_imgs_a[:, 0:self.opt.num_mpi_f, : ,:]
        proj_ref_albedo_a = proj_ref_imgs_a[:, self.opt.num_mpi_f:-1, : ,:]
        proj_ref_alphas_a = proj_ref_imgs_a[:, -1:, :, :]

        composite_albedo_a = projector.over_composite(proj_ref_albedo_a, proj_ref_alphas_a, premultiply_alpha=0).unsqueeze(0)

        depth_planes = depths.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        proj_depth_planes_a = depth_planes.repeat(1, 1, proj_ref_imgs_a.size(2), proj_ref_imgs_a.size(3))
        warp_mpi_depth_a = self.create_mpi_depth(proj_depth_planes_a.to(1), proj_ref_alphas_a)

        proj_ref_mask_a = torch.sum(proj_ref_alphas_a > 1e-8, dim=0, keepdim=True)
        proj_ref_mask_a = (proj_ref_mask_a > self.opt.num_mpi_planes*3//4).type(torch.cuda.FloatTensor)

        # render image at viewpoint a, condition on image a
        latent_z_a_rep = latent_z_a.view(latent_z_a.size(0), latent_z_a.size(1), 1, 1).repeat(proj_ref_imgs_a.size(0), 1, proj_ref_imgs_a.size(2), proj_ref_imgs_a.size(3)) #.expand(z.size(0), z.size(1), x.size(2), x.size(3))        
        
        buffer_a = torch.cat([proj_ref_features_a, proj_ref_albedo_a, proj_ref_alphas_a], dim=1)
        proj_ref_inputs_concat_a = torch.cat([buffer_a, latent_z_a_rep], dim=1)

        render_rgb_mpi_a = self.netG.forward(proj_ref_inputs_concat_a)

        composite_rgb_img_a = projector.over_composite(render_rgb_mpi_a, proj_ref_alphas_a, premultiply_alpha=0).unsqueeze(0)

        return composite_rgb_img_a, composite_albedo_a.detach(), warp_mpi_depth_a, proj_ref_mask_a
            

    def infer_rgb_from_mpi_simple(self, ref_imgs, img_a, 
                                  latent_z_a, 
                                  K_ref, K_a, 
                                  R_ar, t_ar, depths):

        proj_ref_imgs_a = self.mpi_render_view(ref_imgs, img_a, R_ar, t_ar, K_ref, K_a, depths)
        # proj_ref_imgs_b = self.mpi_render_view(ref_imgs, img_b, R_br, t_br, K_ref, K_b, depths).to(1)

        proj_ref_features_a = proj_ref_imgs_a[:, 0:self.opt.num_mpi_f, : ,:]
        proj_ref_albedo_a = proj_ref_imgs_a[:, self.opt.num_mpi_f:-1, : ,:]
        proj_ref_alphas_a = proj_ref_imgs_a[:, -1:, :, :]

        composite_albedo_a = projector.over_composite(proj_ref_albedo_a, proj_ref_alphas_a, premultiply_alpha=0).unsqueeze(0)

        depth_planes = depths.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        proj_depth_planes_a = depth_planes.repeat(1, 1, proj_ref_imgs_a.size(2), proj_ref_imgs_a.size(3))
        warp_mpi_depth_a = self.create_mpi_depth(proj_depth_planes_a, proj_ref_alphas_a)

        proj_ref_mask_a = torch.sum(proj_ref_alphas_a > 1e-8, dim=0, keepdim=True)
        proj_ref_mask_a = (proj_ref_mask_a > self.opt.num_mpi_planes*3//4).type(torch.cuda.FloatTensor)

        # render image at viewpoint a, condition on image a
        latent_z_a_rep = latent_z_a.view(latent_z_a.size(0), latent_z_a.size(1), 1, 1).repeat(proj_ref_imgs_a.size(0), 1, proj_ref_imgs_a.size(2), proj_ref_imgs_a.size(3)) #.expand(z.size(0), z.size(1), x.size(2), x.size(3))        
        # latent_z_b_rep = latent_z_b.view(latent_z_b.size(0), latent_z_b.size(1), 1, 1).repeat(proj_ref_imgs_a.size(0), 1, proj_ref_imgs_a.size(2), proj_ref_imgs_a.size(3)) #.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        
        # print('latent_z_a_rep ', latent_z_a_rep.size())
        # sys.exit()

        # if self.opt.premultiply_alpha:
            # buffer_a = torch.cat([proj_ref_features_a, proj_ref_albedo_a * proj_ref_alphas_a, proj_ref_alphas_a], dim=1)
            # proj_ref_inputs_concat_a = torch.cat([buffer_a, latent_z_a_rep], dim=1)
            # proj_ref_inputs_concat_b = torch.cat([buffer_a, latent_z_b_rep], dim=1)

        # else:
        buffer_a = torch.cat([proj_ref_features_a, proj_ref_albedo_a, proj_ref_alphas_a], dim=1)
        proj_ref_inputs_concat_a = torch.cat([buffer_a, latent_z_a_rep], dim=1)
        # proj_ref_inputs_concat_b = torch.cat([buffer_a, latent_z_b_rep], dim=1)

        # print('proj_ref_inputs_concat_a ', proj_ref_inputs_concat_a.size())
        # sys.exit()

        render_rgb_mpi_a = self.netG.forward(proj_ref_inputs_concat_a)
        composite_rgb_img_a = projector.over_composite(render_rgb_mpi_a, proj_ref_alphas_a, premultiply_alpha=0).unsqueeze(0)

        # print('render_rgb_mpi_a ', render_rgb_mpi_a.size())

        # render image at viewpoint a, condition on image b
        # render_rgb_mpi_b = self.netG.forward(proj_ref_inputs_concat_b)
        # composite_rgb_img_ba = projector.over_composite(render_rgb_mpi_b, proj_ref_alphas_a, premultiply_alpha=self.opt.premultiply_alpha).unsqueeze(0)

        return composite_rgb_img_a, composite_albedo_a.detach(), warp_mpi_depth_a, proj_ref_mask_a


    def render_mpi_imgs_func(self, img_a_full, warp_img_a, img_a, 
                             K_a, R_ar, t_ar):

        composite_feature_ref = projector.over_composite(self.ref_feature_img_small.detach(), self.ref_albedo_rgba_mpi_small[:, -1:, :, :], premultiply_alpha=0).unsqueeze(0)

        warp_concat = torch.cat([warp_img_a, self.composite_albedo_ref, composite_feature_ref], dim=1)

        latent_z_a = self.netE.forward(warp_concat, img_a_full)

        ref_mpi_concat = torch.cat([self.ref_feature_img, self.ref_albedo_mpi], dim=1)

        return self.infer_rgb_from_mpi(ref_mpi_concat, img_a,
                                       latent_z_a,
                                       self.K_ref, K_a, 
                                       R_ar, t_ar, self.mpi_planes)


    def forward(self):  
        K_a = Variable(self.targets['K_a'].cuda(), requires_grad=False)
        K_b = Variable(self.targets['K_b'].cuda(), requires_grad=False)

        R_ar = Variable(self.targets['R_ar'].cuda(), requires_grad=False)
        t_ar = Variable(self.targets['t_ar'].cuda(), requires_grad=False)

        self.img_a_full = Variable(self.targets['img_a_full'].cuda(1), requires_grad=False)
        self.warp_img_a = Variable(self.targets['warp_img_a'].cuda(1), requires_grad=False)
        self.img_a = Variable(self.targets['img_a'].cuda(1), requires_grad=False)

        self.pred_render_rgb_a, self.pred_albedo_a, \
        self.warp_mpi_depth_a, self.proj_ref_mask_a = self.render_mpi_imgs_func(self.img_a_full, self.warp_img_a, self.img_a,
                                                                                K_a, R_ar, t_ar)


    def get_image_paths(self):
        return self.image_paths

    # def write_summary(self, mode_name, 
    #                   img_a, img_a_full, warp_img,
    #                   pred_render_rgb_a, pred_albedo_a,
    #                   warp_mpi_depth_a,
    #                   gt_ref_mask_a,
    #                   loss_dict, n_iter):

    #     if loss_dict is not None:
    #         for loss_dict_key in loss_dict:
    #             self.writer.add_scalar(mode_name + '/' + loss_dict_key, loss_dict[loss_dict_key], n_iter)

    #     pred_render_rgb_a_mask = pred_render_rgb_a * gt_ref_mask_a
    #     # pred_render_rgb_ba_mask = pred_render_rgb_ba * gt_ref_mask_a
    #     # pred_render_rgb_bab = pred_render_rgb_bab * gt_ref_mask_b

    #     self.writer.add_image(mode_name + '/img_a', vutils.make_grid(img_a.data.cpu(), normalize=True), n_iter)
    #     self.writer.add_image(mode_name + '/warp_img', vutils.make_grid(warp_img.data.cpu(), normalize=True), n_iter)
    #     self.writer.add_image(mode_name + '/img_a_full', vutils.make_grid(img_a_full.data.cpu(), normalize=True), n_iter)
    #     # self.writer.add_image(mode_name + '/img_b_full', vutils.make_grid(img_b_full.data.cpu(), normalize=True), n_iter)

    #     self.writer.add_image(mode_name + '/pred_render_rgb_a', vutils.make_grid(pred_render_rgb_a.data.cpu(), normalize=True), n_iter)
    #     # self.writer.add_image(mode_name + '/pred_render_rgb_ba', vutils.make_grid(pred_render_rgb_ba.data.cpu(), normalize=True), n_iter)

    #     self.writer.add_image(mode_name + '/pred_render_rgb_a_mask', vutils.make_grid(pred_render_rgb_a_mask.data.cpu(), normalize=True), n_iter)
    #     # self.writer.add_image(mode_name + '/pred_render_rgb_ba_mask', vutils.make_grid(pred_render_rgb_ba_mask.data.cpu(), normalize=True), n_iter)

    #     self.writer.add_image(mode_name + '/warp_mpi_depth_a', vutils.make_grid(warp_mpi_depth_a.repeat(1, 3, 1, 1).data.cpu(), normalize=True), n_iter)
    #     self.writer.add_image(mode_name + '/pred_albedo_a', vutils.make_grid(pred_albedo_a.data.cpu(), normalize=True), n_iter)

    # def dis_update(self, gt_ref_mask_a):
    #     pred_render_rgb_a = self.pred_render_rgb_a.detach() * gt_ref_mask_a
    #     # pred_render_rgb_ba = self.pred_render_rgb_ba.detach() * gt_ref_mask_a

    #     img_a = self.img_a * gt_ref_mask_a
    #     # img_b = self.img_b * gt_ref_mask_b

    #     loss_dis_a = self.opt.w_gan * self.discriminator.calc_dis_loss(pred_render_rgb_a, img_a, gt_ref_mask_a, gt_ref_mask_a)
    #     # loss_dis_ba = self.opt.w_gan * self.discriminator.calc_dis_loss(pred_render_rgb_ba, img_b, gt_ref_mask_a, gt_ref_mask_b)

    #     return (loss_dis_a)

    def gen_update(self, gt_ref_mask_a):
        pred_render_rgb_a = self.pred_render_rgb_a * gt_ref_mask_a
        # pred_render_rgb_ba = self.pred_render_rgb_ba * gt_ref_mask_a

        loss_gen_a = self.opt.w_gan * self.discriminator.calc_gen_loss(pred_render_rgb_a, gt_ref_mask_a)
        # loss_gen_ba = self.opt.w_gan * self.discriminator.calc_gen_loss(pred_render_rgb_ba, gt_ref_mask_a)

        return (loss_gen_a)

    def compute_style_loss(self, vgg, img_fake, img_real):
        def vgg_preprocess(batch):
            mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).unsqueeze(0)
            std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).unsqueeze(0)

            return (batch - mean) / std

        def compute_gram_matrix(input):
            a, b, c, d = input.size()  # a=batch size(=1)
   
            features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
            G = torch.mm(features, features.t())  # compute the gram product

            # we 'normalize' the values of the gram matrix
            # by dividing by the number of element in each feature maps.
            return G.div(a* b * c * d)

        vgg_fake_img = vgg_preprocess(img_fake)
        vgg_real_img = vgg_preprocess(img_real)

        fake_style = vgg(vgg_fake_img)
        real_style = vgg(vgg_real_img)
        style_loss = 0.

        for x, y in zip(fake_style, real_style):
            gram_fake = compute_gram_matrix(x)
            gram_real = compute_gram_matrix(y)
            style_loss += torch.mean(torch.abs(gram_real - gram_fake))

        return style_loss


    def optimize_parameters(self, n_iter):
        self.forward()

        gt_mask_a = Variable(self.targets['gt_mask_a'].cuda(1), requires_grad=False).unsqueeze(1)
        gt_ref_mask_a = self.proj_ref_mask_a * gt_mask_a

        style_loss = self.opt.style_w * self.compute_style_loss(self.vgg, self.pred_render_rgb_a * gt_ref_mask_a, self.img_a * gt_ref_mask_a)

        loss_g = 0.
        self.optimizer_G.zero_grad()
        loss_dict = {}

        if self.opt.use_gan:
            self.optimizer_D.zero_grad()
            loss_dis_adv = self.dis_update(gt_ref_mask_a)
            print('loss_dis_adv ', loss_dis_adv.item())
            loss_dis_adv.backward()
            self.optimizer_D.step()

            loss_gen_adv = self.gen_update(gt_ref_mask_a)
            print('loss_gen_adv ', loss_gen_adv.item())
            loss_g += loss_gen_adv

            loss_dict['loss_dis_adv'] = loss_dis_adv.item()
            loss_dict['loss_gen_adv'] = loss_gen_adv.item()

        if self.opt.use_vgg_loss:
            loss_img_recon = self.opt.recon_x_w * self.criterion_joint.compute_Perceptual_Loss(self.pred_render_rgb_a, self.img_a, gt_ref_mask_a)

            loss_g += loss_img_recon + style_loss 

            print('loss_vgg_recon ', loss_img_recon.item(), 'style_loss', style_loss.item())

            loss_dict['loss_img_recon'] = loss_img_recon.item()        
            loss_dict['style_loss'] = style_loss.item()
            # loss_dict['loss_gradient'] = loss_gradient.item()

        else:
            loss_img_recon = self.opt.recon_x_w * self.criterion_joint.compute_mae_loss(self.pred_render_rgb_a, self.img_a, gt_ref_mask_a)

            loss_gradient = 0.
            for j in range(self.num_scales):
                stride = 2**j
                loss_gradient += self.opt.w_grad * self.opt.recon_x_w * self.criterion_joint.compute_gradient_loss(self.pred_render_rgb_a[:, :, ::stride, ::stride], 
                                                                                                                   self.img_a[:, :, ::stride, ::stride], 
                                                                                                                   gt_ref_mask_a[:, :, ::stride, ::stride])

            loss_g += loss_img_recon + loss_gradient + style_loss 

            print('loss_img_recon ', loss_img_recon.item(), 
                  'loss_gradient ', loss_gradient.item(), 
                  'style_loss', style_loss.item())


            loss_dict['loss_img_recon'] = loss_img_recon.item()        
            loss_dict['style_loss'] = style_loss.item()
            loss_dict['loss_gradient'] = loss_gradient.item()

        loss_g.backward()
        self.optimizer_G.step()

        torch.cuda.empty_cache()

        if n_iter%300 == 0:
            with torch.no_grad():
                self.write_summary('Train', 
                                   self.img_a, self.img_a_full, self.warp_img_a,
                                   self.pred_render_rgb_a, self.pred_albedo_a,
                                   self.warp_mpi_depth_a,
                                   gt_ref_mask_a,
                                   loss_dict, n_iter)


    def switch_to_train(self):
        self.netE.train()
        self.netG.train()

    def switch_to_eval(self):
        self.netE.eval()
        self.netG.eval()

    def save(self, label):
        self.save_network(self.netG, 'G', label, [1, 2, 3])
        self.save_network(self.netE, 'E', label, [1])
        np.save(self.opt.local_dir + label + 'ref_feature_img.npy', self.ref_feature_img.data.cpu().numpy())

    def update_learning_rate(self):

        self.scheduler.step()
        # for scheduler in self.schedulers:
            # scheduler.step()
        lr = self.optimizer_G.param_groups[0]['lr']
        print('Current learning rate = %.4f' % lr)





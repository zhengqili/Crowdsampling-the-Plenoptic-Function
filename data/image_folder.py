################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################
# import h5py
import torch.utils.data as data
import pickle
import PIL
import numpy as np
import torch
from PIL import Image, ImageEnhance
import os
import math, random
import os.path
import sys, traceback
import cv2
import json
from skimage.transform import rotate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_dataset(list_name):
    text_file = open(list_name, "r")
    images_list = text_file.readlines()
    text_file.close()
    return images_list

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


class LandmarksFolder(data.Dataset):

    def __init__(self, opt, data_dir, phase, img_a_name, img_b_name, img_c_name):

        json_path = data_dir + 'subset_sphere_0_data_clean.json'

        with open(json_path) as json_file:
            self.json_data = json.load(json_file)
        
        self.opt = opt
        self.phase = phase
        self.img_a_name = img_a_name
        self.img_b_name = img_b_name
        self.img_c_name = img_c_name

        num_valid = len(self.json_data)
        num_train = int(round(num_valid * 0.85))

        ref_depth = 1.25
        self.crop_size = 256

        json_data_sub = self.json_data
        near_plane_depth_list = []

        ref_idx = 0
        for i in range(len(json_data_sub)):
            near_plane_depth_list.append(json_data_sub[i]['near_plane_depth'])

        final_depth = np.percentile(np.array(near_plane_depth_list), 10)

        self.scene_scale = ref_depth/final_depth

        ref_list = json_data_sub

        self.mpi_train_list = json_data_sub[0:num_train]
        self.mpi_test_list = json_data_sub[num_train:]

        if phase == 'train':
            self.mpi_list = self.mpi_train_list
        elif phase == 'interpolation':
            self.mpi_list = json_data_sub
        elif phase == 'test':
            self.mpi_list = self.mpi_test_list
        else:
            print('PHASE DOES NOT EXIST')
            sys.exit()

        if opt.dataset == 'trevi':
            scene_id = 36
        elif opt.dataset == 'pantheon':
            scene_id = 23
        elif opt.dataset == 'coeur':
            scene_id = 13
        elif opt.dataset == 'rushmore':
            scene_id = 1589
        elif opt.dataset == 'lincoln':
            scene_id = 21
        elif opt.dataset == 'eiffel':
            scene_id = 0
        elif opt.dataset == 'rock':
            scene_id = 11
        elif opt.dataset == 'navona':
            scene_id = 57

        self.data_dir = data_dir

        self.aspect_ratio_threshold_arr = np.array([9./16., 2./3., 3./4., 1., 4./3., 3./2., 16./9.])

        self.resized_wh = np.array([[288, 512], [320, 480], [384, 512], [384, 384], [512, 384], [480, 320], [512, 288]])
        self.resized_wh_e = np.array([[288, 512], [256, 384], [288, 384], [384, 384], [384, 288], [384, 256], [512, 288]])

        R_rg = np.array(ref_list[ref_idx]['R_cg'])
        t_rg = np.array(ref_list[ref_idx]['t_cg']) 

        self.T_rg = np.eye(4)
        self.T_rg[:3, :3] = R_rg
        self.T_rg[:3, 3] = t_rg * self.scene_scale

    def load_data(self, json_data_sub):
        T_sg = np.eye(4)
        T_sg[:3, :3] = np.array(json_data_sub['R_cg'])
        T_sg[:3, 3] = np.array(json_data_sub['t_cg']) * self.scene_scale

        T_sr = np.dot(T_sg, np.linalg.inv(self.T_rg))
        R_sr = T_sr[:3, :3]
        t_sr = T_sr[:3, 3:4] 

        img_path = os.path.join(self.data_dir, 'images', json_data_sub['img_name'])
        semantic_path = os.path.join(self.data_dir, 'semantics_rotate', json_data_sub['img_name'].replace('.jpg', '.png'))

        warp_path = os.path.join(self.data_dir, 'warp_imgs', json_data_sub['img_name'])

        try:
            warp_img = cv2.imread(warp_path)
            warp_img = np.float32(warp_img[:, :, ::-1])/255.0
            warp_img = cv2.resize(warp_img, (512, 512), interpolation=cv2.INTER_AREA)
        except:
            print('warp_path ', warp_path)
            sys.exit()

        img = cv2.imread(img_path)
        img = np.float32(img[:, :, ::-1])/255.0
        aspect_ratio = float(img.shape[1])/float(img.shape[0])

        if os.path.exists(semantic_path):
            semantic_seg = cv2.imread(semantic_path)
            gt_mask = 1.0 - np.float32(semantic_seg[:, :, 0])/255.0
        else:
            gt_mask = np.ones_like(img[:, :, 0])            

        roll_path = os.path.join(self.data_dir, 'angles', json_data_sub['img_name'].replace('.jpg', '.txt'))

        with open(roll_path, 'r') as f:
            lines = f.readlines()[0].strip()
            roll = float(lines)

        if np.abs(roll- 90.) < 15.:
            gt_mask = rotate(gt_mask, 90, resize=True)
        elif np.abs(roll + 90.) < 15.:
            gt_mask = rotate(gt_mask, -90, resize=True)
        elif np.abs(np.abs(roll) - 180.) < 15.:
            gt_mask = rotate(gt_mask, 180, resize=True)

        K_src = np.array(json_data_sub['K'])

        best_idx = np.argmin(np.abs(aspect_ratio - self.aspect_ratio_threshold_arr))
        resized_width, resized_height = self.resized_wh[best_idx][0], self.resized_wh[best_idx][1]

        resized_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        gt_mask = cv2.resize(gt_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

        scale_x = float(resized_img.shape[1])/float(img.shape[1])
        scale_y = float(resized_img.shape[0])/float(img.shape[0])

        K_src[0, 0] = K_src[0, 0] * scale_x
        K_src[0, 2] = K_src[0, 2] * scale_x
        K_src[1, 1] = K_src[1, 1] * scale_y
        K_src[1, 2] = K_src[1, 2] * scale_y

        resized_width_e, resized_height_e = self.resized_wh_e[best_idx][0], self.resized_wh_e[best_idx][1]
        resized_img_e = cv2.resize(img, (resized_width_e, resized_height_e), interpolation=cv2.INTER_AREA)

        if self.phase == 'train':
            start_y = random.randint(0, resized_height - self.crop_size)
            start_x = random.randint(0, resized_width - self.crop_size)

            # target_img = resized_img
            # # crop image as well
            target_img = resized_img[start_y:start_y+self.crop_size, start_x:start_x+self.crop_size, :]
            gt_mask = gt_mask[start_y:start_y+self.crop_size, start_x:start_x+self.crop_size]

            K_src[0, 2] = K_src[0, 2] - start_x #* scale_x
            K_src[1, 2] = K_src[1, 2] - start_y #* scale_y
        else:
            target_img = resized_img

        return resized_img_e, warp_img, target_img, gt_mask, K_src, R_sr, t_sr

    def __getitem__(self, index):
        targets = {}

        if self.phase == 'interpolation':
            for i in range(len(self.mpi_list)):
                json_data_sub = self.mpi_list[i]

                if self.img_a_name == json_data_sub['img_name']:
                    index_a = i
                
                if self.img_b_name == json_data_sub['img_name']:
                    index_b = i

                if self.img_c_name == json_data_sub['img_name']:
                    index_c = i

            json_data_a = self.mpi_list[index_a]
            json_data_b = self.mpi_list[index_b]
            json_data_c = self.mpi_list[index_c]

            img_c_full, warp_img_c, img_c, gt_mask_c, K_c, R_cr, t_cr = self.load_data(json_data_c)
            # Image C is target viewpoint 
            targets['K_c'] = torch.from_numpy(np.ascontiguousarray(K_c)).contiguous().float()
            targets['R_cr'] = torch.from_numpy(np.ascontiguousarray(R_cr)).contiguous().float()
            targets['t_cr'] = torch.from_numpy(np.ascontiguousarray(t_cr)).contiguous().float()

            targets['img_c_full'] = torch.from_numpy(np.ascontiguousarray(img_c_full).transpose(2,0,1)).contiguous().float()
            targets['warp_img_c'] = torch.from_numpy(np.ascontiguousarray(warp_img_c).transpose(2,0,1)).contiguous().float()

            targets['img_c'] = torch.from_numpy(np.ascontiguousarray(img_c).transpose(2,0,1)).contiguous().float()
            targets['gt_mask_c'] = torch.from_numpy(np.ascontiguousarray(gt_mask_c)).contiguous().float()
            targets['img_path_c'] = json_data_c['img_name']

        else:
            json_data_a = self.mpi_list[index]
            json_data_b = self.mpi_list[random.randint(0, len(self.mpi_list)-1)]

        img_a_full, warp_img_a, img_a, gt_mask_a, K_a, R_ar, t_ar = self.load_data(json_data_a)
        img_b_full, warp_img_b, img_b, gt_mask_b, K_b, R_br, t_br = self.load_data(json_data_b)

        # plt.figure(figsize=(12, 8))
        # plt.subplot(2,3,1)
        # plt.imshow(img_a_full) 

        # plt.subplot(2,3,2)
        # plt.imshow(warp_img_a)

        # plt.subplot(2,3,3)
        # plt.imshow(img_a)

        # plt.subplot(2,3,4)
        # plt.imshow(img_b_full)

        # plt.subplot(2,3,5)
        # plt.imshow(warp_img_b)

        # plt.subplot(2,3,6)
        # plt.imshow(img_b)

        # plt.savefig(json_data_a['img_name'])

        # print('%s data loader, we are good'%self.opt.dataset)
        # sys.exit()  

        # final_img = None #torch.from_numpy(np.ascontiguousarray(target_img).transpose(2,0,1)).contiguous().float()

        # targets['K_ref'] = torch.from_numpy(np.ascontiguousarray(self.K_ref)).contiguous().float()
        targets['K_a'] = torch.from_numpy(np.ascontiguousarray(K_a)).contiguous().float()
        targets['K_b'] = torch.from_numpy(np.ascontiguousarray(K_b)).contiguous().float()

        targets['R_ar'] = torch.from_numpy(np.ascontiguousarray(R_ar)).contiguous().float()
        targets['t_ar'] = torch.from_numpy(np.ascontiguousarray(t_ar)).contiguous().float()

        targets['R_br'] = torch.from_numpy(np.ascontiguousarray(R_br)).contiguous().float()
        targets['t_br'] = torch.from_numpy(np.ascontiguousarray(t_br)).contiguous().float()

        targets['img_a_full'] = torch.from_numpy(np.ascontiguousarray(img_a_full).transpose(2,0,1)).contiguous().float()
        targets['img_b_full'] = torch.from_numpy(np.ascontiguousarray(img_b_full).transpose(2,0,1)).contiguous().float()

        targets['warp_img_a'] = torch.from_numpy(np.ascontiguousarray(warp_img_a).transpose(2,0,1)).contiguous().float()
        targets['warp_img_b'] = torch.from_numpy(np.ascontiguousarray(warp_img_b).transpose(2,0,1)).contiguous().float()

        targets['img_a'] = torch.from_numpy(np.ascontiguousarray(img_a).transpose(2,0,1)).contiguous().float()
        targets['img_b'] = torch.from_numpy(np.ascontiguousarray(img_b).transpose(2,0,1)).contiguous().float()

        targets['gt_mask_a'] = torch.from_numpy(np.ascontiguousarray(gt_mask_a)).contiguous().float()
        targets['gt_mask_b'] = torch.from_numpy(np.ascontiguousarray(gt_mask_b)).contiguous().float()

        targets['img_path_a'] = json_data_a['img_name']
        targets['img_path_b'] = json_data_b['img_name']
        
        return targets

    def __len__(self):
        return len(self.mpi_list)


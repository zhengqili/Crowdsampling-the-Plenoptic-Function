from __future__ import division
import numpy as np
from torch.autograd import Variable
import torch
# import homography
import sys


def fwd_homography(K_ref, K_tgt, R_tr, t_tr, n_hat, d):
	"""Computes inverse homography matrix between two cameras via a plane.
	Args:
	  K_ref: intrinsics for ref cameras, [..., 3, 3] matrices
	  K_tgt: intrinsics for target cameras, [..., 3, 3] matrices
	  rot: relative rotations from reference to target, [..., 3, 3] matrices
	  t: [..., 3, 1], translations from reference to target camera. Mapping a 3D
		point p from reference to target is accomplished via rot * p + t.
	  n_hat: [..., 1, 3], plane normal w.r.t reference camera frame
	  a: [..., 1, 1], plane equation displacement
	Returns:
	  homography: [..., 3, 3] inverse homography matrices (homographies mapping
		pixel coordinates from target to ref).
	"""
	# print('K_tgt ', K_tgt, K_tgt.size())
	K_tgt_inv = torch.inverse(K_tgt)

	H = R_tr + 1./d * torch.bmm(t_tr, n_hat)

	fwd_h = torch.bmm(torch.bmm(K_ref, H), K_tgt_inv)

	return fwd_h



def projective_forward_homography(ref_imgs, K_ref, tgt_imgs, K_tgt, R_tr, t_tr, depths, cuda_id=0):
	'''
		warp reference image to target image using planes in the reference image
	'''

	def planar_transform(ref_imgs, tgt_imgs, pixel_coords_tgt, 
						 K_ref, K_tgt, R_tr, t_tr, n_hat, d):

		num_planes, num_c, ref_height, ref_width = ref_imgs.size()
		tgt_height, tgt_width = tgt_imgs.size(2), tgt_imgs.size(3)

		# print('fwd_homography(K_ref, K_tgt, R_tr, t_tr, n_hat, d) ', fwd_homography(K_ref, K_tgt, R_tr, t_tr, n_hat, d))
		# sys.exit()

		hom_t2r_planes = torch.inverse(fwd_homography(K_ref, K_tgt, R_tr, t_tr, n_hat, d))

		pixel_coords_t2r = hom_t2r_planes.bmm(pixel_coords_tgt)

		z_2 = pixel_coords_t2r[:, 2:3, :].clamp(min=1e-8).repeat(1, 2, 1)  
		xy_warp = pixel_coords_t2r[:, 0:2 ,:]/z_2  

		x_norm = xy_warp[:, 0, :]/float(ref_width-1)
		y_norm = xy_warp[:, 1, :]/float(ref_height-1)

		flow_field = torch.stack((x_norm, y_norm), dim=-1)

		flow_field = flow_field.view(num_planes, tgt_height, tgt_width, 2)

		imgs_r2t = torch.nn.functional.grid_sample(ref_imgs, 2.*flow_field - 1., mode='bilinear', padding_mode='zeros')

		return imgs_r2t


	num_planes, num_c, ref_height, ref_width = ref_imgs.size()
	tgt_height, tgt_width = tgt_imgs.size(2), tgt_imgs.size(3)

	xx, yy = np.meshgrid(range(tgt_width), range(tgt_height))
	xx = np.ravel(xx)
	yy = np.ravel(yy)
	ones = np.ones_like(xx)
	xy_h = np.tile(np.expand_dims(np.stack((xx ,yy, ones), axis=0), axis=0), (num_planes, 1, 1))
	pixel_coords_tgt = Variable(torch.from_numpy(np.ascontiguousarray(xy_h)).float().cuda(cuda_id), requires_grad=False)

	n_hat = Variable(torch.from_numpy(np.ascontiguousarray(np.array([0., 0., 1.]))).contiguous().float().cuda(cuda_id).unsqueeze(0), requires_grad=False)
	n_hat = n_hat.repeat(num_planes, 1, 1)
	d = depths.unsqueeze(-1).unsqueeze(-1)

	proj_ref_images = planar_transform(ref_imgs, tgt_imgs, pixel_coords_tgt, 
									   K_ref.repeat(num_planes, 1, 1), K_tgt.repeat(num_planes, 1, 1), 
									   R_tr.repeat(num_planes, 1, 1), t_tr.repeat(num_planes, 1, 1), n_hat, d)
	
	return proj_ref_images


def projective_inverse_homography(ref_imgs, K_ref, tgt_imgs, K_tgt, R_tr, t_tr, depths):
	'''
		warp reference image to target image using planes in the target image
	'''

	def planar_transform(ref_imgs, tgt_imgs, pixel_coords_tgt, 
						 K_ref, K_tgt, R_tr, t_tr, n_hat, d):

		num_planes, num_c, ref_height, ref_width = ref_imgs.size()
		tgt_height, tgt_width = tgt_imgs.size(2), tgt_imgs.size(3)

		hom_t2r_planes = fwd_homography(K_ref, K_tgt, R_tr, t_tr, n_hat, d)

		pixel_coords_t2r = hom_t2r_planes.bmm(pixel_coords_tgt)

		z_2 = pixel_coords_t2r[:, 2:3, :].clamp(min=1e-8).repeat(1, 2, 1)  
		xy_warp = pixel_coords_t2r[:, 0:2 ,:]/z_2  

		x_norm = xy_warp[:, 0, :]/float(ref_width-1)
		y_norm = xy_warp[:, 1, :]/float(ref_height-1)

		flow_field = torch.stack((x_norm, y_norm), dim=-1)

		flow_field = flow_field.view(num_planes, tgt_height, tgt_width, 2)

		imgs_r2t = torch.nn.functional.grid_sample(ref_imgs, 2.*flow_field - 1., mode='bilinear', padding_mode='zeros')

		return imgs_r2t


	num_planes, num_c, ref_height, ref_width = ref_imgs.size()
	tgt_height, tgt_width = tgt_imgs.size(2), tgt_imgs.size(3)

	xx, yy = np.meshgrid(range(tgt_width), range(tgt_height))
	xx = np.ravel(xx)
	yy = np.ravel(yy)
	ones = np.ones_like(xx)
	xy_h = np.tile(np.expand_dims(np.stack((xx ,yy, ones), axis=0), axis=0), (num_planes, 1, 1))
	pixel_coords_tgt = Variable(torch.from_numpy(np.ascontiguousarray(xy_h)).float().cuda(), requires_grad=False)

	n_hat = Variable(torch.from_numpy(np.ascontiguousarray(np.array([0., 0., 1.]))).contiguous().float().cuda().unsqueeze(0), requires_grad=False)
	n_hat = n_hat.repeat(num_planes, 1, 1)
	d = depths.unsqueeze(-1).unsqueeze(-1)

	proj_ref_images = planar_transform(ref_imgs, tgt_imgs, pixel_coords_tgt, 
									   K_ref.repeat(num_planes, 1, 1), K_tgt.repeat(num_planes, 1, 1), 
									   R_tr.repeat(num_planes, 1, 1), t_tr.repeat(num_planes, 1, 1), n_hat, d)
	
	return proj_ref_images


def over_composite(imgs, alphas, premultiply_alpha=0):
	"""Combines a list of RGBA images using the over operation.

	Combines RGBA images from back to front with the over operation.
	The alpha image of the first image is ignored and assumed to be 1.0.

	Args:
	rgbas: A list of [batch, 4, H, W] RGBA images, combined from back to front.
	Returns:
	Composited RGB image.
	"""
	num_planes = imgs.size(0)
	num_features = imgs.size(1)

	for i in range(num_planes):
		rgb = imgs[i, :, :, :]
		alpha = alphas[i, :, :, :]

		if i == 0:
			output = rgb
		else:
			if premultiply_alpha:
				rgb_by_alpha = rgb
			else:
				rgb_by_alpha = rgb * alpha
			
			output = rgb_by_alpha + output * (1.0 - alpha)

	return output



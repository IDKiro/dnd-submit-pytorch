import os
import random
import torch
import numpy as np
import glob
import PIL.Image as Image
from torch.utils.data import Dataset

from utils import read_img, hwc_to_chw, ISP


def get_patch(origin_img, noise_img, patch_size, random):
	H = origin_img.shape[0]
	W = origin_img.shape[1]

	ps_temp = min(H, W, patch_size + 1) - 1

	if random:
		xx = np.random.randint(0, W-ps_temp)
		yy = np.random.randint(0, H-ps_temp)

		patch_origin_img = origin_img[yy:yy+ps_temp, xx:xx+ps_temp, :]
		patch_noise_img = noise_img[yy:yy+ps_temp, xx:xx+ps_temp, :]

		if np.random.randint(2, size=1)[0] == 1:
			patch_origin_img = np.flip(patch_origin_img, axis=1).copy()
			patch_noise_img = np.flip(patch_noise_img, axis=1).copy()
		if np.random.randint(2, size=1)[0] == 1: 
			patch_origin_img = np.flip(patch_origin_img, axis=0).copy()
			patch_noise_img = np.flip(patch_noise_img, axis=0).copy()
		if np.random.randint(2, size=1)[0] == 1:
			patch_origin_img = np.transpose(patch_origin_img, (1, 0, 2)).copy()
			patch_noise_img = np.transpose(patch_noise_img, (1, 0, 2)).copy()

	else:
		xx = int(W / 2 - ps_temp / 2)
		yy = int(H / 2 - ps_temp / 2)

		patch_origin_img = origin_img[yy:yy+ps_temp, xx:xx+ps_temp, :]
		patch_noise_img = noise_img[yy:yy+ps_temp, xx:xx+ps_temp, :]

	return patch_origin_img, patch_noise_img


class RealNoise(Dataset):
	def __init__(self, root_dir, patch_size=128, random=True):
		self.root_dir = root_dir
		self.batches = sorted(os.listdir(self.root_dir))
		self.patch_size = patch_size
		self.origin_imgs = [None] * len(self.batches)
		self.noise_imgs = [None] * len(self.batches)
		self.random = random

		for i in range(len(self.batches)):
			self.origin_imgs[i] = []
			self.noise_imgs[i] = []

	def __len__(self):
		l = len(self.batches)
		return l

	def __getitem__(self, idx):
		batch_path = os.path.join(self.root_dir, self.batches[idx])

		if not len(self.origin_imgs[idx]):
			origin_fns = glob.glob(batch_path + '/Reference.png')
			noise_fns = glob.glob(batch_path + '/Noisy.png')

			self.origin_imgs[idx] = read_img(origin_fns[0])

			for noise_fn in noise_fns:
				self.noise_imgs[idx].append(read_img(noise_fn))
				
		origin_img = self.origin_imgs[idx]
		noise_img = self.noise_imgs[idx][np.random.randint(len(self.noise_imgs[idx]))]

		patch_origin_img, patch_noise_img = get_patch(origin_img, noise_img, self.patch_size, self.random)

		patch_origin_img_chw = hwc_to_chw(patch_origin_img)
		patch_noise_img_chw = hwc_to_chw(patch_noise_img)

		return patch_noise_img_chw, patch_origin_img_chw


class SynNoise(Dataset):
	def __init__(self, root_dir, sigma_s=[], sigma_c=[], patch_size=128, random=True):
		self.root_dir = root_dir
		self.filenames = glob.glob(root_dir + '/*.png')
		self.patch_size = patch_size
		self.origin_imgs = [None] * len(self.filenames)
		self.noise_imgs = [None] * len(self.filenames)

		self.sigma_s = sigma_s
		self.sigma_c = sigma_c

		self.isp = ISP(curve_path='./utils/noise/')

		self.random = random

		for i in range(len(self.filenames)):
			self.origin_imgs[i] = []
			self.noise_imgs[i] = []

	def __len__(self):
		l = len(self.filenames)
		return l

	def __getitem__(self, idx):
		if not len(self.origin_imgs[idx]):
			origin_img = read_img(self.filenames[idx])

			if not (len(self.sigma_s) and len(self.sigma_c)):
				# all random
				min_log = np.log([0.000001])
				sigma_s = min_log + np.random.rand(1) * (np.log([0.002]) - min_log)
				sigma_s = np.exp(sigma_s)
				sigma_c = min_log + np.random.rand(1) * (np.log([0.001]) - min_log)
				sigma_c = np.exp(sigma_c)
			else:
				sigma_s = self.sigma_s
				sigma_c = self.sigma_c

			self.origin_imgs[idx], self.noise_imgs[idx] = self.isp.cbdnet_noise_generate_srgb(origin_img, sigma_s, sigma_c)

		origin_img = self.origin_imgs[idx]
		noise_img = self.noise_imgs[idx]

		patch_origin_img, patch_noise_img = get_patch(origin_img, noise_img, self.patch_size, self.random)

		patch_origin_img_chw = hwc_to_chw(patch_origin_img)
		patch_noise_img_chw = hwc_to_chw(patch_noise_img)

		return patch_noise_img_chw, patch_origin_img_chw


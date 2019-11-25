from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil, importlib
import numpy as np
import argparse
import torch
import torch.nn as nn
from skimage import io

from utils import AverageMeter, chw_to_hwc
from dataset.loader import RealNoise

parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('--model', default='resnet', type=str, help='model name')
parser.add_argument('--ps', default=512, type=int, help='patch size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=1000, type=int, help='sum of epochs')
parser.add_argument('--freq', default=500, type=int, help='learning rate update frequency')
parser.add_argument('--save_freq', default=100, type=int, help='save result frequency')
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch):
	if not epoch % args.freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
	return optimizer


def train(train_loader, model, criterion, optimizer, epoch, result_dir):
	losses = AverageMeter()
	model.train()

	for ind, (noise_img, origin_img) in enumerate(train_loader):
		st = time.time()

		input_var = noise_img.cuda()
		target_var = origin_img.cuda()

		output = model(input_var)
		loss = criterion(output, target_var)

		losses.update(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print('[{0}][{1}]\t'
			'lr: {lr:.5f}\t'
			'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
			'Time: {time:.3f}'.format(
			epoch, ind,
			lr=optimizer.param_groups[-1]['lr'],
			loss=losses,
			time=time.time()-st))

		if epoch % args.save_freq == 0:
			if not os.path.isdir(os.path.join(result_dir, '%04d'%epoch)):
				os.makedirs(os.path.join(result_dir, '%04d'%epoch))

			origin_np = origin_img.numpy()
			noise_np = noise_img.numpy()
			output_np = output.cpu().detach().numpy()

			origin_np_img = chw_to_hwc(origin_np[0])
			noise_np_img = chw_to_hwc(noise_np[0])
			output_img = chw_to_hwc(np.clip(output_np[0], 0, 1))

			temp = np.concatenate((origin_np_img, noise_np_img, output_img), axis=1)
			io.imsave(os.path.join(result_dir, '%04d/train_%d.jpg'%(epoch, ind)), np.uint8(temp * 255))


if __name__ == '__main__':

	train_dir = './data/real/'
	save_dir = os.path.join('./save_model/', args.model)
	result_dir = os.path.join('./result/', args.model)

	model = importlib.import_module('.' + args.model, package='model').Network()
	print(model)
	model.cuda()

	if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
		# load existing model
		model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
		print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']
	else:
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		# create model
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		cur_epoch = 0

	criterion = nn.MSELoss()
	criterion.cuda()

	train_dataset = RealNoise(train_dir, patch_size=args.ps)
	
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=1, shuffle=True, pin_memory=True)

	for epoch in range(cur_epoch, args.epochs + 1):

		optimizer = adjust_learning_rate(optimizer, epoch)
		train(train_loader, model, criterion, optimizer, epoch, result_dir)

		torch.save({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()}, 
			os.path.join(save_dir, 'checkpoint.pth.tar'))
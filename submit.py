from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil, importlib
import torch
import argparse

from utils import pytorch_denoiser, denoise_srgb, bundle_submissions_srgb
from model.resnet import Network

parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('--model', default='resnet', type=str, help='model name')
args = parser.parse_args()

input_dir = './data/test/'
output_dir = './result/test/'
save_dir = os.path.join('./save_model/', args.model)

model = importlib.import_module('.' + args.model, package='model').Network()
print(model)
model.cuda()
denoiser = pytorch_denoiser(model, True)

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
    print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
else:
    print('Error: no trained model detected!')
    exit(1)

with torch.no_grad():
    denoise_srgb(denoiser, input_dir, output_dir)

bundle_submissions_srgb(output_dir)
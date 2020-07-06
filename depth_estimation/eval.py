import os
import argparse
import time
import numpy as np
import cv2
from PIL import Image
import sys
import tkinter as tk

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from bts import *
# import utils
import glob

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(args, file):
    if file.endswith('.npy') == 1:
        img = np.load(file)
        img = img.squeeze() if img.ndim == 3 else img         
        img = cv2.resize(img, (args.width, args.height), interpolation = cv2.INTER_NEAREST)
    else:
        img = cv2.resize(cv2.imread(file), (args.width, args.height), interpolation = cv2.INTER_NEAREST) [...,::-1]
    return img

def load_focal(args):
    if args.dataset == 'nyu': 
        focal = Variable(torch.tensor([518.8579])).to(device)  
    elif args.dataset == 'kitti':
        focal = Variable(torch.tensor([715.0873])).to(device) 
    elif args.dataset == 'diode':
        focal = Variable(torch.tensor([886.81])).to(device)
    return focal
    
def preprocess(dm, validity_mask):
    masker = np.zeros_like(validity_mask)
    masker[validity_mask==1] = 1
    MIN_DEPTH = 0.0
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm_clip = np.clip(dm, MIN_DEPTH, MAX_DEPTH)    
    
    dm_clip[masker==0] = 0
    
    return dm_clip

def previous_pre(dm, validity_mask):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.0
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm_clip = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    gt_depth = np.where(validity_mask, dm_clip, 0) * 1000

    return gt_depth

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def show(img, cmap='jet'):
    
    plt.imshow(img, cmap = cmap); plt.show()

def predict(image, focal):
    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    tensor_img = to_tensor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(tensor_img, focal) 
    pred_depth = depth_est.cpu().numpy().squeeze()
    
    return pred_depth

def load_model(args):
    model = BtsModel(args)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    return model

def parser(params):
 
    parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
    parser.add_argument('--model_name', type=str, help='model name', default=r'diode_resnet_diode')
    parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or resnet50_bts',
                        default='resnet50_bts')
    parser.add_argument('--checkpoint', type=str, help='path to a specific checkpoint to load' )
    parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='diode')
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation' )
    parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
    parser.add_argument('--height', type=int, help='input height', default=768) # 256 x 416 , 416 x 544 , 256 x 320
    parser.add_argument('--width', type=int, help='input width', default=1024) 
    parser.add_argument('--folder_path' , default = '/home/zmh/hdd/Projects/Group-A/Depth_Map/dataset/diode_val/val' , help= 'evaluation folder path') 
    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=100)

    return parser.parse_args(params) 
    
if __name__ == '__main__':
    
    params = [
        
        '--max_depth', '100',
        '--checkpoint', '/home/teama/dev/src/gitlab/bts/pytorch/checkpoints/bts_diode_resnet_pre_4/100.0-model-28647-best_rms_5.91021', 
        '--folder_path', '/home/teama/dev/src/gitlab/Datasets/diode_depth/val'
        ]
    args = parser(params)
    model_dir = os.path.dirname(args.checkpoint)
    sys.path.append(model_dir)  
    model = load_model(args)
    focal = load_focal(args)
    
    indoor_path = args.folder_path + '/indoor'
    outdoor_path = args.folder_path + '/outdoor'
    
    files = {}
    indoor = []
    #indoor files
    for filename in glob.iglob(indoor_path + '**/**/*.png', recursive=True):
        indoor.append(filename)
        
    outdoor = []
    #outdoor files
    for filename in glob.iglob(outdoor_path + '**/**/*.png', recursive=True):
        outdoor.append(filename)
    
    files['indoor'] = indoor
    files['outdoor'] = outdoor
    eval_measure = np.zeros(10)
    for i in range(files['indoor'].__len__()):
        
        depth_file = os.path.join(files['indoor'][i][:-4] + '_depth.npy')
        mask_file = os.path.join(files['indoor'][i][:-4] + '_depth_mask.npy')
        
        image = load_image(args, files['indoor'][i])
        gt_depth = load_image(args, depth_file)
        gt_mask = load_image(args, mask_file)
        gt_image = preprocess(gt_depth, gt_mask)
        
        pred_depth = predict(image, focal)
        # mask = gt_image != 0
        
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval) 
        
        measures = compute_errors(gt_image[valid_mask], pred_depth[valid_mask])
        eval_measure[:9] += measures
        eval_measure[-1] += 1
    
    cnt = eval_measure[-1]
    print('Computing errors for {} eval samples'.format(int(cnt)))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                 'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                 'd3'))
 
        
        
        
        
        
        
        
        
 
    
        


    
    

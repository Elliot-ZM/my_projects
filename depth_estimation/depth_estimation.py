import cv2
from PIL import Image
import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms 
from bts import *
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class DepthEstimation:
    def __init__(self, args):
        self.args = args
        
    def preprocess(self, image, gt_depth, gt_mask):
        assert isinstance(image, np.ndarray) , 'image must be np array'
         
        if self.dataset == 'diode':
            focal = Variable(torch.tensor([886.81])).to(device)
            if isinstance(gt_depth, np.ndarray):
                gt_depth = utils.add_mask(gt_depth, gt_mask)
                gt_depth = gt_depth * 1000
                gt_depth = cv2.resize(gt_depth, (self.args.width, self.args.height), cv2.INTER_NEAREST)
            else: 
                gt_depth = None

        elif self.dataset == 'realsense':
            focal = Variable(torch.tensor([886.81])).to(device)
            if isinstance(gt_depth, np.ndarray):
                gt_depth[gt_depth > 9000] = 0
                gt_depth = gt_depth[45:472, 43:608] # crop dude to depth image pixel registration
                gt_depth = cv2.resize(gt_depth, (self.args.width, self.args.height), cv2.INTER_NEAREST)
            else: 
                gt_depth = None
            image = image[45:472, 43:608]
            
        elif self.dataset == 'nyu':
            focal = Variable(torch.tensor([518.8579])).to(device)
            if isinstance(gt_depth, np.ndarray):
                gt_depth = gt_depth[45:472, 43:608]
                gt_depth = cv2.resize(gt_depth, (self.args.width, self.args.height), cv2.INTER_NEAREST)
            else: 
                gt_depth = None
            
        elif self.dataset == 'kitti':
            focal = Variable(torch.tensor([715.0873])).to(device)
    
        image = cv2.resize(image, (self.args.width, self.args.height), cv2.INTER_NEAREST)            
        return image, gt_depth, focal
    
    def post_process(self, pred_depth):
        if self.dataset == 'kitti':
            pred_depth = pred_depth.cpu().numpy().squeeze() * 256
        else:
            pred_depth = pred_depth.cpu().numpy().squeeze() * 1000
        colormap = cv2.applyColorMap(cv2.convertScaleAbs(pred_depth, alpha=0.0355), cv2.COLORMAP_JET)[...,::-1]

        return pred_depth, colormap
   
    def predict(self, model, image, gt_depth, dataset = 'realsense', gt_mask= None):
        self.dataset = dataset
        image, gt_depth, focal = self.preprocess(image, gt_depth, gt_mask)
        to_tensor = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.3947, 0.3610, 0.3366], [0.1965, 0.1943, 0.2006])
        ])
        tensor_img = to_tensor(Image.fromarray(image)).unsqueeze(0).to(device)
        *_, pred_depth = model(tensor_img, focal)
        pred_depth, colormap = self.post_process(pred_depth)
        
        return pred_depth, colormap
        
    def load_model(self, args):    
        model_dir = os.path.dirname(args.model_path)
        sys.path.append(model_dir)
        model = BtsModel(args)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.to(device)
        
        return model
        
        
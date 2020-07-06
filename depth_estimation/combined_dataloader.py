import os, glob 
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import torch
import albumentations as A
from distributed_sampler_no_evenly_divisible import *
import torch.utils.data.distributed 
from visualization_utils import compare_plot

class Combined_Dataset(Dataset):
    def __init__(self, args, mode, transform = False):
        self.root = args.data_root
        self.mode = mode
        self.args = args
        imgs = []
        
        if self.mode == 'train':
            imgs = glob.glob(os.path.join(self.root, 'train') + '**/**/rgb*.jpg', recursive = True)
            # diode (outdoor) + rs
            for img in glob.glob(os.path.join(self.root,'train') + '**/**/*.png', recursive = True):
                if not 'sync' in img:
                    imgs.append(img)
                    
        elif self.mode == 'test':
            imgs = glob.glob(os.path.join(self.root, 'val') + '**/**/rgb*.jpg', recursive = True)
            # diode (outdoor) + rs
            for img in glob.glob(os.path.join(self.root,'val') + '**/**/*.png', recursive = True):
                if not 'sync' in img:
                    imgs.append(img)
        
        else:
            print('Unsupported mode. Try with [train or test]')
            return -1
        
        self.imgs = imgs
        self.transform = transform
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        focal = 886.81
        if self.imgs[index].endswith('.jpg'): # nyu
           self.dataset = 'nyu'
           rgb_path = self.imgs[index]
           depth_path = rgb_path.replace('rgb_', 'sync_depth_')[:-4]+'.png'
           mask_path = None
        
        elif 'outdoor' in self.imgs[index]: # diode
            self.dataset = 'diode'
            rgb_path = self.imgs[index]
            depth_path = rgb_path[:-4] + '_depth.npy'
            mask_path = rgb_path[:-4] + '_depth_mask.npy'  
        
        elif not 'outdoor' in self.imgs[index]: # realsense
            self.dataset = 'realsense'
            rgb_path = self.imgs[index]
            depth_path = rgb_path[:-4] + '_depth.npy'
            mask_path = None
        
        image, depth_gt = self.load_data(rgb_path, depth_path, mask_path)
        image, depth_gt = self.resize(image, depth_gt)
        
        if self.mode == 'train': 
            depth_gt = np.expand_dims(depth_gt, axis = 2)
            if self.transform:
                sample = self.augment(image, depth_gt, p = 0.4)
        
        elif self.mode == 'test':
            depth_gt = np.expand_dims(depth_gt, axis = 2)
            sample = {'image': image, 'depth': depth_gt}

        image = self.to_tensor(sample['image']).type(torch.float32)
        depth_gt = torch.from_numpy(sample['depth'].transpose(2, 0, 1))
        # print(index, '--> ',image.shape, depth_gt.shape)
        return {'image': image, 'depth': depth_gt, 'focal': focal}
        
    def load_data(self, rgb_path, depth_path, mask_path):
        image = Image.open(rgb_path).convert('RGB')
        image = np.array(image, dtype = np.uint8)
        
        if self.dataset == 'diode':
            depth_gt = np.load(depth_path).squeeze()
            depth_mask = np.load(mask_path)
            depth_gt = self.add_mask(depth_gt, depth_mask)
            
        elif self.dataset == 'realsense':
            depth_gt = (np.load(depth_path) / 1000).astype('float32')
            depth_gt = depth_gt[45:472, 43:608]
            depth_gt[depth_gt > 9] = 0
            image = image[45:472, 43:608]
  
        elif self.dataset == 'nyu':
            depth_gt = Image.open(depth_path)
            depth_gt = np.asarray(depth_gt, dtype = np.float32) / 1000
            depth_gt = depth_gt[45:472, 43:608]
            image = image[45:472, 43:608]
     
        return image, depth_gt
    
    def resize(self, image, depth):
        image = cv2.resize(image, (self.args.input_width, self.args.input_height), interpolation = cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.args.input_width, self.args.input_height), interpolation = cv2.INTER_NEAREST)
        return image, depth
    
    def add_mask(self, depth_gt, depth_mask):
        MIN_DEPTH = 0.0
        MAX_DEPTH = min(300, np.percentile(depth_gt, 99))
        depth_gt = np.clip(depth_gt, MIN_DEPTH, MAX_DEPTH) 
        depth_gt[depth_mask==0] = 0
        return depth_gt 

    def augment(self, image, depth_gt, p = 0.3):
        augs = A.Compose([
            A.RandomGamma(gamma_limit = (80, 140)),
            A.RandomBrightnessContrast(),
        ], p = 0.3)
        
        image = self.augmenter(augs, image)
        # same augmentations with rgb and depth image
        sample = self.sync_augmenter([
            A.HorizontalFlip(),
    #         A.ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.2, rotate_limit = 45)
        ], 
            [image, depth_gt], p)
        return sample
    
    def augmenter(self, aug, img):
        return aug(image = img)['image']
    
    def sync_augmenter(self, aug, images, p):
        target = {'depth': 'image'}
        sample = A.Compose(aug, p = p, additional_targets= target)(image = images[0], depth = images[1])
        return sample

class Combined_Dataloader(DataLoader):
    def __init__(self, args, mode, debug = False):
                
        if mode == 'train':
            self.training_samples = Combined_Dataset(args, mode, transform =  True)
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None     
            
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   pin_memory=True,
                                   sampler=self.train_sampler,
                                   num_workers=args.num_threads,
                                   ) 
            if debug:
                sample = next(iter(self.data))
                for i in range(len(sample['image'])):
                    img = self.to_numpy(sample['image'][i])
                    depth = self.to_numpy(sample['depth'][i], cmap= 'jet')
                    compare_plot(figsize = (15,15), cmap='jet', image = img, depth = depth)
        
        elif mode == 'test':
            self.testing_samples = Combined_Dataset(args, mode, transform= False)
            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
                
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)     
            if debug:
                sample = next(iter(self.data))
                for i in range(len(sample['image'])):
                    img = self.to_numpy(sample['image'][i])
                    depth = self.to_numpy(sample['depth'][i], cmap= 'jet')
                    compare_plot(figsize = (15,15), cmap='jet', image = img, depth = depth)
    
    def to_numpy(self, img, cmap = 'gray'):
        if img.shape[0] == 3:
            img = img.permute(1,2,0).numpy()
            
        elif img.shape[0] == 1:
            img = img.squeeze().numpy()
            
        return img
        
    
if __name__ == '__main__':
    import argparse 
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, help= 'dataset directory', default= '/home/teama/dev/src/gitlab/Datasets/diode+nyu+rs')
    ap.add_argument('--input_height', type = int, help = 'height', default =416)
    ap.add_argument('--input_width', type = int, help = 'width', default = 544)
    ap.add_argument('--distributed', action = 'store_true', help = 'multi gpu')
    ap.add_argument('--batch_size' , help = 'bs', default= 24)
    ap.add_argument('--num_threads', help= 'worker', default =0)
    args = ap.parse_args()
    
    # train_loader = Combined_Dataloader(args, mode = 'train' , debug= True)
    test_loader = Combined_Dataloader(args, mode = 'test', debug= True)
    
    # for sample in train_loader.data:
        # image , label = sample['image'], sample['depth']
 
  

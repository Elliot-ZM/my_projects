import os
import glob
import cv2
import numpy as np
from PIL import Image

def depth2color(img): 
    color = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.0355), cv2.COLORMAP_JET)[:,:,::-1]    
    return color   

def add_mask(depth, validity_mask): 
    MIN_DEPTH = 0.0
    MAX_DEPTH = min(300, np.percentile(depth, 99))
    depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH) 
    depth[ validity_mask== 0] = 0 
    return depth

def fetch_data(imgs, index, dataset = 'diode'):
    if dataset == 'nyu': # nyu
           rgb_path = imgs[index]
           depth_path = rgb_path.replace('rgb_', 'sync_depth_')[:-4]+'.png'
           mask_path = None
        
    elif dataset == 'realsense': # realsense  
        rgb_path = imgs[index]
        depth_path = rgb_path[:-4] + '_depth.npy'
        mask_path = None
    
    elif dataset == 'diode' : # diode
        rgb_path = imgs[index]
        depth_path = rgb_path[:-4] + '_depth.npy'
        mask_path = rgb_path[:-4] + '_depth_mask.npy'
        
    return rgb_path, depth_path, mask_path

def resize(image, depth, size):
    image = cv2.resize(image, size, interpolation = cv2.INTER_NEAREST)
    depth = cv2.resize(depth, size, interpolation = cv2.INTER_NEAREST)
    return image, depth

def load_data(rgb_path, depth_path, mask_path, dataset = 'diode'):
    image = Image.open(rgb_path).convert('RGB')
    image = np.array(image, dtype = np.uint8)
    
    if dataset == 'diode':
        depth_gt = np.load(depth_path).squeeze()
        depth_mask = np.load(mask_path)
        depth_gt = add_mask(depth_gt, depth_mask)
        
    elif dataset == 'realsense':
        depth_gt = (np.load(depth_path) / 1000).astype('float32')
        # depth_gt = depth_gt[45:472, 43:608]
        depth_gt[depth_gt > 10] = 0
        # image = image[45:472, 43:608]
  
    elif dataset == 'nyu':
        depth_gt = Image.open(depth_path)
        depth_gt = np.asarray(depth_gt, dtype = np.float32) / 1000
        # depth_gt = depth_gt[45:472, 43:608]
        # image = image[45:472, 43:608]
 
    return image, depth_gt

def make_video(img_folder, save_name, fps= 15, size = (640,480) , suffix = '', dataset = 'diode'):
    #fetch data
    images = [os.path.join(img_folder, img) for img in sorted(os.listdir(img_folder)) if img.endswith((suffix))]
    images.sort(key=os.path.getctime)
    ## nyu
    # images = [img for img in sorted(glob.glob(img_folder + f'**/{suffix}/rgb*', recursive= True))]

    # frame = cv2.imread(os.path.join(img_folder, images[0])) 
    # h, w , c = frame.shape
    
    if dataset == 'kitti':
        out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (size[0], size[1]*2))
    else:
        out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (size[0]*2, size[1]))
        
    for index in range(len(images)):
        rgb_path, depth_path, mask_path = fetch_data(images, index, dataset= dataset)
        image, depth = load_data(rgb_path, depth_path, mask_path, dataset= dataset)
        image, depth = resize(image, depth, size = size)
        color_map = depth2color(depth*1000)
        if dataset == 'kitti':
            concat_img = cv2.vconcat([image, color_map])[...,::-1]
            out.write(concat_img)
        else:
            concat_img = cv2.hconcat([image, color_map])[...,::-1]  
            out.write(concat_img)
    
    out.release()
    print('Finished {}'.format(save_name))
 
def check_video(save_name):
    cap = cv2.VideoCapture(save_name)
    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.destroyAllWindows()
            break
            
        cv2.imshow('Check_Video', frame)
        if cv2.waitKey(30) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()   
    
if __name__ == '__main__':
    
    img_folder = '/home/teama/dev/src/gitlab/Datasets/realsense_jp/extract_data_fill_2/test/sugino3'
    out_path = '/home/teama/dev/src/gitlab/Datasets/Videos_from_Datasets'
    dataset = 'realsense'
    save_name = os.path.join(out_path, dataset, f'{dataset}_sugino3.avi')
    make_video(img_folder, save_name, fps= 45, size = (640,480), suffix = '.png', dataset = dataset)    
    
    check_video(save_name)

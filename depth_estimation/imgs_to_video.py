import os
import glob
import cv2
import numpy as np

def make_video(img_folder, save_name, fps= 15, suffix = ''):
    #fetch data
    images = [img for img in sorted(os.listdir(img_folder)) if img.endswith((suffix))]
    ## nyu
    # images = [img for img in sorted(glob.glob(img_folder + f'**/{suffix}/rgb*', recursive= True))]
    # images = [img for img in sorted(glob.glob(img_folder + '**/**/rgb*', recursive= True))]
    frame = cv2.imread(os.path.join(img_folder, images[0]))
    h, w , c = frame.shape
    out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
    
    for img in images:
        out.write(cv2.imread(os.path.join(img_folder, img)))
    
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
    
    img_folder = '/home/teama/dev/src/gitlab/Datasets/nyu_depth_v2/sync'
    
    out_path = '/home/teama/dev/src/gitlab/Datasets/Videos_from_Datasets'
    
    save_name = os.path.join(out_path, 'nyu_classroom.avi')
    make_video(img_folder, save_name, fps= 20, suffix = 'classroom*')    
    
    check_video(save_name)

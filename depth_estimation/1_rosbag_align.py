import cv2                               
import numpy as np                      
import matplotlib.pyplot as plt         
import pyrealsense2 as rs            
import os
import argparse

def main(args):
    out_path = os.path.join(args.bag_file[:-4])
    os.makedirs(out_path , exist_ok= True)
    try:
        pipeline =rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, args.bag_file, False)
    
        pipeline.start(config)
        for x in range(5):
            pipeline.wait_for_frames()
        i = 0 
        align = rs.align(rs.stream.color)
         
        while True:
            print('Saving Frame: ', i)
            frame_present, frameset = pipeline.try_wait_for_frames()
            if not frame_present:
                break
            frameset = align.process(frameset)
            # colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
            depth_frame = frameset.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            cv2.imwrite(os.path.join(out_path , 'Depth_'+ os.path.splitext(os.path.basename(args.bag_file))[0] +'_'+ str(i)+'.png'), depth_image)
            
            color_frame = frameset.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imwrite(os.path.join(out_path, 'Color_'+os.path.splitext(os.path.basename(args.bag_file))[0] +'_'+ str(i)+'.png'), color_image[...,::-1])
            i += 1
            
    except RuntimeError:
        print("No more frames arrived, reached end of BAG file!")
    
    finally:
        pass

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag_file', type = str, help= 'input bag file', deafult= '/home/teama/dev/src/gitlab/Datasets/realsense_jp/higuchi3.bag') 
    args = ap.parse_args()
    main(args)














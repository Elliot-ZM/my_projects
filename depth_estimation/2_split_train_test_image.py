import shutil
import os
import numpy as np
import argparse
import glob
def get_files_from_folder(path):

    files = [i for i in os.listdir(path) if i.startswith('Color')]
    return np.asarray(files)

def main(path_to_data, path_to_test_data, train_ratio):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))
   
    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    test_counter = np.round(data_counter_per_class * (1 - train_ratio))

    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])
        path_to_save = os.path.join(path_to_test_data, dirs[i])

        #creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst1 = os.path.join(path_to_save, files[j])
            dst2 = os.path.join(path_to_save, files[j].replace('Color','Depth'))
            src1 = os.path.join(path_to_original, files[j])
            src2 = os.path.join(path_to_original, files[j].replace('Color','Depth'))
            shutil.move(src1, dst1)
            shutil.move(src2, dst2)
 
def parse_args():
  parser = argparse.ArgumentParser(description="Dataset divider")
  parser.add_argument("--data_path", help="Path to data",
                      default = '/home/zmh/hdd/Projects/Group-A/Depth_Map/dataset/realsense_jp/train')
  parser.add_argument("--test_data_path_to_save", help="Path to test data where to save",
                      default = '/home/zmh/hdd/Projects/Group-A/Depth_Map/dataset/realsense_jp/val')
  parser.add_argument("--train_ratio", type = float , help="Train ratio - 0.7 means splitting data in 70 % train and 30 % test",
                      default = 0.85)
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  main(args.data_path, args.test_data_path_to_save, float(args.train_ratio))
import numpy as np

def load_video_info(path):
    gt_path = path + '/groundtruth_rect.txt'
    ground_truth = np.loadtxt(gt_path)

    

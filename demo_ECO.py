import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
print cur_path
sys.path.append(cur_path + '\\feature_extraction\\')
sys.path.append(cur_path + '\\implementation\\')
sys.path.append(cur_path + '\\runfiles\\')
sys.path.append(cur_path + '\\utils\\')
print sys.path

import load_video_info
import testing_ECO

video_path = 'sequence/Crossing'
seq, ground_truth = load_video_info(video_path)
results = testing_ECO(seq)

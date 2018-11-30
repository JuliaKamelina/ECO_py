import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
print cur_path
sys.path.append(cur_path + '\\feature_extraction\\')
sys.path.append(cur_path + '\\implementation\\')
sys.path.append(cur_path + '\\implementation\\initialization\\')
sys.path.append(cur_path + '\\runfiles\\')
sys.path.append(cur_path + '\\utils\\')
print sys.path

from load_video_info import *
from testing_ECO import *

video_path = "{}\\sequences\\Crossing".format(cur_path)
seq, ground_truth = load_video_info(video_path)
results = testing_ECO(seq)

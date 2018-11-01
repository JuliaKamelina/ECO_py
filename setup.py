import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
print cur_path
sys.path.append(cur_path + '\\feature_extraction\\')
sys.path.append(cur_path + '\\implementation\\')
sys.path.append(cur_path + '\\runfiles\\')
sys.path.append(cur_path + '\\utils\\')
print sys.path

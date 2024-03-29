import sys
import os
import cv2
import numpy as np
import argparse

from PIL import Image
sys.path.append('./')

from implementation import ECOTracker
from implementation.utils import load_video_info, get_sequence_info


def demo_tracker(video_path, no_show):
    seq, ground_truth = load_video_info(video_path)
    seq = get_sequence_info(seq)
    frames = [np.array(Image.open(f)) for f in seq["image_files"]]
    is_color = True if (len(frames[0].shape) == 3) else False
    tracker = ECOTracker(seq, frames[0], is_color)
    for i, frame in enumerate(frames):
        if i == 0:
            output = tracker.init_tracker(frame)
        else:
            output, _ = tracker.track(frame, i)
        bbox = output.get('target_bbox', seq['init_rect'])
        time = output.get('time', 1)
        print(i)
        print('bb: ', bbox)
        print(time)
        if is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 255), 1)
        gt_bbox = (ground_truth[i, 0],
                   ground_truth[i, 1],
                   ground_truth[i, 0] + ground_truth[i, 2],
                   ground_truth[i, 1] + ground_truth[i, 3])
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0]), int(gt_bbox[1])),
                              (int(gt_bbox[2]), int(gt_bbox[3])),
                              (0, 255, 0), 1)
        print('gt: ', gt_bbox)
        print("#######################################################################")
        if not no_show:
            cv2.imshow('', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="sequences/Car1")
    parser.add_argument("--no_show", action='store_true')
    args = parser.parse_args()
    demo_tracker(args.video, args.no_show)
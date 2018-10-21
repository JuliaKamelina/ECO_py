import load_video_info
import testing_ECO

video_path = 'sequence/Crossing'
seq, ground_truth = load_video_info(video_path)
results = testing_ECO(seq)

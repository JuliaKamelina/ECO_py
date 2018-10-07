import matplotlib.pyplot

def get_sequence_info(**seq):
    if (!('format' in seq.keys())):
        seq['format'] = 'vot'  # TODO: CHECK!!!

    seq['frame'] = 0
    if (seq['format'] == 'otb'):
        seq['init_sz'] = np.array([seq['init_rect'][3], seq['init_rect'][2]])
        seq['init_pos'] = np.array([seq['init_rect'][1], seq['init_rect'][0]]) + (seq.init_sz - 1)/2
        seq['num_frames'] = len(seq['image_files'])
        seq['rect_position'] = zeros((seq['num_frames'], 4))
        init_image = imread(seq['image_files'][0])

    return(seq, init_image)

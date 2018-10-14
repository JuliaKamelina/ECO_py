import numpy as np

def init_features(features, **gparams, is_color_image = False, img_sample_sz = 0, size_mode = ''):
    if (size_mode == ''):
        size_mode = 'same'

    gp_keys = gparams.keys()
    if not 'normalize_power' in gp_keys:
        gparams['normalize_power'] = []
    if not 'normalize_size' in gp_keys:
        gparams['normalize_size'] = True
    if not 'normalize_dim' in gp_keys:
        gparams['normalize_dim'] = False
    if not 'square_root_normalization' in gp_keys:
        gparams['square_root_normalization'] = False
    if not 'use_gpu' in gp_keys:
        gparams['use_gpu'] = False

    keep_features = []
    for i in range(1:len(features)):
        f_keys = features[i]['fparams']
        if not 'useForColor' in f_keys:
            features[i]['fparams']['useForColor'] = True
        if not 'useForGray' in f_keys:
            features[i]['fparams']['useForGray'] = True

        if ((features[i]['fparams']['useForColor'] && is_color_image) ||
            (features[i]['fparams']['useForGray'] && !is_color_image)):
            keep_features.append(features[i])

    features = keep_features
    num_features = len(features)
    feature_info = {}
    feature_info['min_cell_size'] = np.zeros((num_frames, 1))

    for i in range(1:len(features)):
        if 'getFeature' in features[i]

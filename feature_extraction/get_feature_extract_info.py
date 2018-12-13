import numpy as np

def check(feature_img_sz, info_img_sz):
    for i in range(0, len(feature_img_sz)):
        if not feature_img_sz[i] in info_img_sz:
            return False
    return True

def get_feature_extract_info(features):
    extract_info = {}
    extract_info["img_sample_sizes"] = np.array([])
    extract_info["img_input_sizes"] = np.array([])
    for i in range(0, len(features)):
        if not check(features[i]["img_sample_sz"], extract_info["img_sample_sizes"]):
            extract_info["img_sample_sizes"] = np.insert(extract_info["img_sample_sizes"], i, features[i]["img_sample_sz"], axis=0)
            shape = extract_info["img_sample_sizes"].shape
            if len(shape) == 1:
                extract_info["img_sample_sizes"] = extract_info["img_sample_sizes"].reshape(1, shape[0])
            
            extract_info["img_input_sizes"] = np.insert(extract_info["img_input_sizes"], i, features[i]["img_input_sz"], axis=0)
            shape = extract_info["img_input_sizes"].shape
            if len(shape) == 1:
                extract_info["img_input_sizes"] = extract_info["img_input_sizes"].reshape(1, shape[0])
    return(extract_info)

def get_feature_extract_info(features):
    extract_info = {}
    extract_info["img_sample_sizes"] = []
    extract_info["img_input_sizes"] = []
    for i in range(0, len(features)):
        if not features[i]["img_sample_sz"] in extract_info["img_sample_sizes"]:
            extract_info["img_sample_sizes"].append(features[i]["img_sample_sz"])
            extract_info["img_input_sizes"].append(features[i]["img_input_sz"])
    return(extract_info)

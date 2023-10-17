import os
import os.path as p
import yaml

MAIN_PATH = "."
for library in ["mindspore", "paddle", "tensorflow", "torch"]:
    LIBRARY_PATH = p.join(MAIN_PATH, library)
    SIMILARITY_PATH = p.join(LIBRARY_PATH, "api_similarity")
    target_path = p.join(LIBRARY_PATH, "layer_similarity.yaml")
    file_list = os.listdir(SIMILARITY_PATH)
    result_dict = {}
    for file_name in file_list:
        api_name = file_name[:-5]
        f = open(p.join(SIMILARITY_PATH, file_name), "r", encoding="utf-8")
        current_dict = yaml.full_load(f)
        f.close()
        result_dict[api_name] = current_dict
    f2 = open(target_path, "w", encoding="utf-8")
    yaml.dump(result_dict, f2)
    f2.close()

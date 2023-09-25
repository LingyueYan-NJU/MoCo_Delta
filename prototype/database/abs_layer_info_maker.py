import yaml
import os
ABS_LAYER_FORMAT_PATH = "./Abs_Layer_Format"
ABS_LAYER_INFO_PATH = "./Abs_Layer_Info"


def get_layer_info(torch_layer_name: str) -> dict:
    path = "./torch_layer_info"
    yaml_path = os.path.join(path, torch_layer_name + ".yaml")
    f = open(yaml_path, "r", encoding="utf-8")
    d = yaml.full_load(f)
    f.close()
    return d


def get_abs_layer_format(aln: str) -> dict:
    yaml_path = os.path.join(ABS_LAYER_FORMAT_PATH, aln + ".yaml")
    f = open(yaml_path, "r", encoding="utf-8")
    d = yaml.full_load(f)
    f.close()
    return d


def create_empty_full_dict() -> dict:
    result = {"api": None, "constraints": {}, "descp": None, "inputs": {"optional": [], "required": []}}
    return result


def analyse_abs_layer_info_dict(aln: str) -> dict:
    abs_layer_format = get_abs_layer_format(aln)
    torch_layer_info_dict = get_layer_info(abs_layer_format["api"]["torch"])
    valid_para_list = list(abs_layer_format["params"].keys())
    abs_layer_info_dict = create_empty_full_dict()
    abs_layer_info_dict["api"] = aln
    abs_layer_info_dict["descp"] = torch_layer_info_dict["descp"]
    for para in torch_layer_info_dict["constraints"].keys():
        if para in valid_para_list:
            abs_layer_info_dict["constraints"][para] = torch_layer_info_dict["constraints"][para]
    for para in torch_layer_info_dict["inputs"]["optional"]:
        if para in valid_para_list:
            abs_layer_info_dict["inputs"]["optional"].append(para)
    for para in torch_layer_info_dict["inputs"]["required"]:
        if para in valid_para_list:
            abs_layer_info_dict["inputs"]["required"].append(para)
    return abs_layer_info_dict


abs_layer_list = []
temp = os.listdir(ABS_LAYER_FORMAT_PATH)
for abs_file_name in temp:
    abs_layer_list.append(abs_file_name[:-5])
for abs_layer_name in abs_layer_list:
    d = analyse_abs_layer_info_dict(abs_layer_name)
    f = open(os.path.join(ABS_LAYER_INFO_PATH, abs_layer_name + ".yaml"), "w", encoding="utf-8")
    yaml.dump(d, f)
    f.close()
    print(abs_layer_name + " already written.")

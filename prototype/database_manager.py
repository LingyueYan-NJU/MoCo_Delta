import yaml
import os


class DBManager:
    def __init__(self):
        ABS_LAYER_FORMAT_PATH = "./database/Abs_Layer_Format"
        ABS_LAYER_INFO_PATH = "./database/Abs_Layer_Info"
        ABS_LAYER_SIMILARITY_PATH = "./database/Abs_Layer_Similarity"
        self.abs_layer_format_dict = {}
        self.abs_layer_info_dict = {}
        self.abs_layer_similarity_dict = {}
        file_list = os.listdir(ABS_LAYER_FORMAT_PATH)
        self.abs_layer_list = []
        for file in file_list:
            self.abs_layer_list.append(file[:-5])
        for file in file_list:
            name = file[:-5]
            f = open(os.path.join(ABS_LAYER_FORMAT_PATH, file), "r", encoding="utf-8")
            self.abs_layer_format_dict[name] = yaml.full_load(f)
            f.close()
            f = open(os.path.join(ABS_LAYER_INFO_PATH, file), "r", encoding="utf-8")
            self.abs_layer_info_dict[name] = yaml.full_load(f)
            f.close()
            f = open(os.path.join(ABS_LAYER_SIMILARITY_PATH, file), "r", encoding="utf-8")
            self.abs_layer_similarity_dict[name] = yaml.full_load(f)
            f.close()
        return

    def get_api_name(self, abs_api_name: str, lib: str) -> str:
        assert abs_api_name in self.abs_layer_list and lib in ["torch", "paddle"]
        return self.abs_layer_format_dict[abs_api_name]["api"][lib]

    def get_para_name(self, abs_api_name: str, abs_para_name: str, lib: str):
        assert abs_api_name in self.abs_layer_list and lib in ["torch", "paddle"]
        d = self.abs_layer_format_dict[abs_api_name]["params"]
        assert abs_para_name in d.keys()
        return d[abs_para_name][lib]

    def get_api_list(self):
        return self.abs_layer_list

    def is_valid_api(self, abs_api_name: str) -> bool:
        return abs_api_name in self.abs_layer_list

    def get_similarity_dict(self, abs_api_name: str) -> dict:
        assert self.is_valid_api(abs_api_name)
        return self.abs_layer_similarity_dict[abs_api_name]

    def get_layer_info(self, abs_api_name: str) -> dict:
        assert self.is_valid_api(abs_api_name)
        return self.abs_layer_info_dict[abs_api_name]

    def get_para_constraint_dict(self, abs_api_name: str, abs_para_name: str) -> dict:
        assert self.is_valid_api(abs_api_name)
        return self.abs_layer_info_dict[abs_api_name]["constraints"][abs_para_name]

    def get_seed_model(self, abs_model_name: str) -> dict:
        assert abs_model_name in ["LeNet"]
        f = open("./database/seeds/" + abs_model_name + ".yaml")
        result = yaml.full_load(f)
        f.close()
        return result


dbm = DBManager()

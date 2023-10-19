import random
from _database import database
import yaml
import os.path as p


def roulette_wheel_selection(dictionary):
    total_value = sum(dictionary.values())
    random_number = random.uniform(0, total_value)
    current_sum = 0
    for key, value in dictionary.items():
        current_sum += value
        if current_sum >= random_number:
            return key


class Mutator:
    def __init__(self):
        self.mode = 0
        self.library_list = []
        self.threshold = 0.0
        self.refresh_config()
        return

    def refresh_config(self) -> None:
        CONFIG_PATH = p.join("..", "config", "config.yaml")
        f = open(CONFIG_PATH, "r", encoding="utf-8")
        file_config = yaml.full_load(f)
        f.close()
        self.library_list = list(file_config["LIBRARY_LIST"].values())
        self.mode = file_config["MODE"]
        self.threshold = file_config["THRESHOLD"]
        return

    def mutate(self, abstract_layer_dict: dict) -> (dict, str):
        abstract_layer_name = abstract_layer_dict["layer"]
        abstract_layer_info = database.get_abstract_layer_info(abstract_layer_name)
        if random.choice([1, 2]) == 1:
            result, mutate_info = self.api_name_mutate(abstract_layer_dict, abstract_layer_info)
        else:
            result, mutate_info = self.api_para_mutate(abstract_layer_dict, abstract_layer_info)
        return result, mutate_info

    def api_name_mutate(self, layer_dict: dict, abstract_layer_info: dict) -> (dict, str):
        abstract_layer_name = layer_dict["layer"]
        if self.mode == 2:
            if database.is_abstract_api_name_valid(abstract_layer_name):
                candidate_list = database.get_candidate_mutate_list(abstract_layer_name)
            else:
                return layer_dict, "no mutate"
            if len(candidate_list) > 0:
                new_api_name = random.choice(candidate_list)
            else:
                return layer_dict, "no mutate"

            # new parameter adoption process
            para = {}
            # TODO
            # new parameter adoption process

            result = {"layer": new_api_name, "params": {}, "in": layer_dict["in"], "out": layer_dict["out"]}
            return result, new_api_name
        else:
            return layer_dict, ""
        # elif self.mode == 1:
        #     # choose api
        #     implicit_api_name = database.get_implicit_api_name(self.library_list[0], abstract_layer_name)
        #     similarity = database.get_implicit_api_similarity_valid(self.library_list[0], implicit_api_name)
        #     if len(similarity) == 0:
        #         return layer_dict, "no mutate"
        #     new_implicit_api_name = roulette_wheel_selection(similarity)
        #     new_abstract_api_name = database.get_abstract_api_name(self.library_list[0], new_implicit_api_name)
        #     mutate_info = new_abstract_api_name
        #     # choose api
        #
        #     # add context similarity here
        #
        #     # new parameter adoption process
        #     new_para_dict = {}
        #     new_layer_info = database.get_abstract_layer_info(new_abstract_api_name)
        #     required_list = new_layer_info["inputs"]["required"]
        #     params_constraint_dict = new_layer_info["constraints"]
        #     para_dict = layer_dict["params"]
        #     for pp in para_dict.items():
        #         if pp[0] in params_constraint_dict.keys():
        #             new_para_dict[pp[0]] = pp[1]
        #     for param_name in required_list:
        #         if param_name not in new_para_dict.keys():
        #             new_para_dict[param_name] = get_value(params_constraint_dict[param_name])[0]
        #
        #     # new parameter adoption process
        #     result = {"layer": new_abstract_api_name, "params": new_para_dict, "in": layer_dict["in"],
        #     "out": layer_dict["out"]}
        #     return result, mutate_info

    def api_para_mutate(self, layer_dict: dict, abstract_layer_info: dict) -> (dict, str):
        if self.mode == 2:
            # TODO
            return self.api_name_mutate(layer_dict, abstract_layer_info)[0], "para mutate"
        else:
            return layer_dict, ""
        # elif self.mode == 1:
        #     # TODO
        #     return self.api_name_mutate(layer_dict, abstract_layer_info)


mutator = Mutator()

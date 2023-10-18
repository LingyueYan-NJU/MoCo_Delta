import random
from _database import database
import yaml
import os.path as p


def roulette_wheel_selection(prob_dict):
    sum_prob = sum(prob_dict.values())
    rand = random.uniform(0, 1)
    proportion_list = [prob / sum_prob for prob in prob_dict.values()]
    dist_list = [abs(proportion_list[i] - rand) for i in range(len(proportion_list))]
    min_dist_index = dist_list.index(min(dist_list))
    return list(prob_dict.keys())[min_dist_index]


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

    def mutate(self, abstract_layer_dict: dict) -> (dict, dict):
        mutate_info = {}
        abstract_layer_name = abstract_layer_dict["layer"]
        abstract_layer_info = database.get_abstract_layer_info(abstract_layer_name)
        if random.choice([1, 2]) == 1:
            result = self.api_name_mutate(abstract_layer_dict, abstract_layer_info)
        else:
            result = self.api_para_mutate(abstract_layer_dict, abstract_layer_info)
        if abstract_layer_dict["layer"] == result["layer"]:
            mutate_info["mutate type"] = "no mutate"
        else:
            mutate_info["mutate type"] = "mutated"
        return result, mutate_info

    def api_name_mutate(self, layer_dict: dict, abstract_layer_info: dict) -> dict:
        abstract_layer_name = layer_dict["layer"]
        if self.mode == 2:
            if database.is_abstract_api_name_valid(abstract_layer_name):
                candidate_list = database.get_candidate_mutate_list(abstract_layer_name)
            else:
                return layer_dict
            if len(candidate_list) > 0:
                new_api_name = random.choice(candidate_list)
            else:
                return layer_dict

            # new parameter adoption process
            para = {}
            # TODO
            # new parameter adoption process

            result = {"layer": new_api_name, "params": {}, "in": layer_dict["in"], "out": layer_dict["out"]}
            return result
        elif self.mode == 1:
            # choose api
            implicit_api_name = database.get_implicit_api_name(self.library_list[0], abstract_layer_name)
            similarity = database.get_api_similarity(self.library_list[0], implicit_api_name)
            candidate_dict = {}
            for key in similarity.keys():
                if similarity[key] >= self.threshold:
                    candidate_dict[key] = similarity[key]
            if len(candidate_dict) == 0:
                return layer_dict
            new_implicit_api_name = roulette_wheel_selection(candidate_dict)
            new_abstract_api_name = database.get_abstract_api_name(self.library_list[0], new_implicit_api_name)
            # choose api

            # add context similarity here

            # new parameter adoption process
            params = {}
            new_layer_info = database.get_abstract_layer_info(new_abstract_api_name)

            # new parameter adoption process
            result = {"layer": new_abstract_api_name, "params": params, "in": layer_dict["in"], "out": layer_dict["out"]}
            return result

    def api_para_mutate(self, layer_dict: dict, abstract_layer_info: dict) -> dict:
        if self.mode == 2:
            # TODO
            return self.api_name_mutate(layer_dict, abstract_layer_info)
        elif self.mode == 1:
            # TODO
            return self.api_name_mutate(layer_dict, abstract_layer_info)


mutator = Mutator()
# test
layer = {'layer': 'maxpool2d', 'params': {}, 'in': 'x', 'out': 'x'}
for i in range(100):
    print(mutator.mutate(layer))

import random
from _database import database


def mutate(layer_dict: dict) -> (dict, dict):
    mutate_info = {}
    if random.choice([1, 2]) == 1:
        result = api_name_mutate(layer_dict)
    else:
        result = api_para_mutate(layer_dict)
    if layer_dict["layer"] == result["layer"]:
        mutate_info["mutate type"] = "no mutate"
    else:
        mutate_info["mutate type"] = "mutated"
    return result, mutate_info


def api_name_mutate(layer_dict: dict) -> dict:
    layer_name = layer_dict["layer"]
    if database.is_abstract_api_name_valid(layer_name):
        candidate_list = database.get_candidate_mutate_list(layer_name)
    else:
        return layer_dict
    if len(candidate_list) > 0:
        new_api_name = random.choice(candidate_list)
    else:
        return layer_dict

    # new parameter adoption process
    para = {}
    # new parameter adoption process

    result = {"layer": new_api_name, "params": {}, "in": layer_dict["in"], "out": layer_dict["out"]}
    return result


def api_para_mutate(layer_dict: dict) -> dict:
    return api_name_mutate(layer_dict)

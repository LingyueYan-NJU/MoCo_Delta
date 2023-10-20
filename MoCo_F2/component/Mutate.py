from abc import ABC, abstractmethod
import random
from _database import database


def roulette_wheel_selection(dictionary):
    total_value = sum(dictionary.values())
    random_number = random.uniform(0, total_value)
    current_sum = 0
    for key, value in dictionary.items():
        current_sum += value
        if current_sum >= random_number:
            return key


class Mutator(ABC):
    def __init__(self):
        return

    @abstractmethod
    def mutate(self, layer_dict: dict) -> (dict, str):
        # 随机选择api替换和参数替换，然后返回变异后的新layer_dict以及变异信息（用于剪枝）
        # 注意，此处传入的都是抽象层面的layer_dict
        pass

    @abstractmethod
    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    @abstractmethod
    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    @abstractmethod
    def get_value(self, para_constraint_dict: dict) -> str:
        # 根据一个标准的约束字典，获取一个值，用于替换后参数适配和参数变异
        pass


class TorchMutator(Mutator):
    def mutate(self, layer_dict: dict) -> (dict, str):
        return layer_dict, "hei_hei"

    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    def get_value(self, para_constraint_dict: dict) -> str:
        pass


class JittorMutator(Mutator):
    def mutate(self, layer_dict: dict) -> (dict, str):
        return layer_dict, "hei_hei"

    def api_name_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    def api_para_mutate(self, layer_dict: dict) -> (dict, str):
        pass

    def get_value(self, para_constraint_dict: dict) -> str:
        pass


def get_mutator(library: str) -> Mutator | None:
    if library == "torch":
        return TorchMutator()
    elif library == "jittor":
        return JittorMutator()
    else:
        return None

import os.path as p
import time
import yaml
import os
import random
from abc import ABC, abstractmethod
from _database import database


def generate_line(api_name: str, params_dict: dict) -> str:
    new_params: str = ""
    for _ in params_dict:
        new_params = new_params + _ + "=" + params_dict[_].__str__() + ", "

    new_params = new_params[:-2]
    new_params = "(" + new_params + ")"

    return api_name + new_params


class Performer(ABC):
    def __init__(self):
        self.at = 1
        return

    @abstractmethod
    def get_library_name(self) -> str:
        pass

    @abstractmethod
    def translate(self, abstract_model: dict) -> str:
        # Given a dict of abstract model, translate it into a str which can be written into a .py file
        # And then this .py file will be a whole model.
        pass

    @abstractmethod
    def get_model_from_file(self, file_path: str):
        # Given a file path of a .py file, turn the net in this file to a model in memory.
        # Return this model.
        pass

    @abstractmethod
    def train(self, model) -> float:
        # Given a model, train it, and return time cost(if failed, return -1).
        pass

    @abstractmethod
    def run(self, model) -> (float, float, list[int], str):
        # Given a model, give a tensor and run it, then
        # 1. return time cost(-1 if failed).
        # 2. return the calculate result(-1 if failed).
        # 3. return the shape([] if failed).
        # 4. return the error info("" if succeeded).
        pass


class TorchPerformer(Performer):
    # TODO
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "torch"

    def translate(self, abstract_model: dict) -> str:
        head = "import torch\nimport torch.nn as nn\n\n\n"
        body = ""
        model_name_list = list(abstract_model.keys())
        for model in abstract_model:
            body += self.__dict_to_model_class(abstract_model[model], model, model_name_list)
        return head + body

    def get_model_from_file(self, file_path: str):
        return "torch model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, float, list[int], str):
        return random.choice([1.0, 2.0, 3.0, -1]), 1.0, [1, 1, 1, 1], "error?"

    def __dict_to_model_class(self, model_dict: dict, model_name: str, model_name_list: list[str]) -> str:
        def_part = "class " + model_name + "(nn.Module):\n    def __init__(self):\n        super("\
                   + model_name + ", self).__init__()\n"
        forward_part = "    def forward(self, x):\n"
        # layer_name = "layer"
        layer_index = 0
        if isinstance(model_dict[0], list):
            _model_dict = model_dict[1:]
            extra = ", ".join(model_dict[0])
            def_part = "class " + model_name + "(nn.Module):\n    def __init__(self, " + extra + ")" +\
                       ":\n        super("+ model_name + ", self).__init__()\n"
        else:
            _model_dict = model_dict
        for layer_dict in _model_dict:
            layer_index += 1
            abstract_layer_name = layer_dict["layer"]
            if abstract_layer_name == "cat":
                forward_part_line = "        " + layer_dict["out"] + " = torch.cat("
                if isinstance(layer_dict["in"], list):
                    forward_part_line += str(layer_dict["in"]).replace("'", "")
                else:
                    forward_part_line += layer_dict["in"]
                forward_part_line += ", dim=" + str(layer_dict["params"]["dims"]) + ")\n"
                forward_part += forward_part_line
            elif abstract_layer_name in model_name_list:
                implicit_layer_name = abstract_layer_name
                implicit_params = layer_dict["params"]
                def_part_line = "        self." + "layer" + str(layer_index) + " = " +\
                                generate_line(implicit_layer_name, implicit_params) + "\n"
                forward_part_line = "        " + layer_dict["out"] + " = " + "self.layer" + str(layer_index) + "(" +\
                                    layer_dict["in"] + ")\n"
                def_part += def_part_line
                forward_part += forward_part_line
            else:
                implicit_layer_name = database.get_implicit_api_name(self.get_library_name(), abstract_layer_name)
                implicit_params = {}
                abstract_params = layer_dict["params"]
                for abstract_param_name in abstract_params.keys():
                    implicit_param_name = database.get_implicit_para_name(self.get_library_name(),
                                                                          abstract_layer_name, abstract_param_name)
                    implicit_params[implicit_param_name] = abstract_params[abstract_param_name]
                def_part_line = "        self." + "layer" + str(layer_index) + " = " +\
                                generate_line(implicit_layer_name, implicit_params) + "\n"
                forward_part_line = "        " + layer_dict["out"] + " = " + "self.layer" + str(layer_index) + "(" +\
                                    layer_dict["in"] + ")\n"
                def_part += def_part_line
                forward_part += forward_part_line
        return def_part + "\n" + forward_part + "        return x\n\n"


class JittorPerformer(Performer):
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "jittor"

    def translate(self, abstract_model: dict) -> str:
        return "jittor model code"

    def get_model_from_file(self, file_path: str):
        return "jittor model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, float, list[int], str):
        return 1.0, 1.0, [1, 1, 1, 1], ""


class TensorFlowPerformer(Performer):
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "tensorflow"

    def translate(self, abstract_model: dict) -> str:
        return "tensorflow model code"

    def get_model_from_file(self, file_path: str):
        return "tensorflow model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, float, list[int], str):
        return 1.0, 1.0, [1, 1, 1, 1], ""


def translator_factory(library: str) -> Performer | None:
    if library == "torch":
        return TorchPerformer()
    elif library == "tensorflow":
        return TensorFlowPerformer()
    elif library == "jittor":
        return JittorPerformer()
    else:
        return None


class Concrete:
    def __init__(self):
        self.__experiment_id = "experiment" + str(time.time())
        print("experiment id: " + self.__experiment_id)
        RESULT_PATH = p.join("..", "result")
        os.makedirs(p.join(RESULT_PATH, self.__experiment_id))
        self.__model_name = ""
        self.__library_list: list[str] = []
        self.__performer_list: list[Performer] = []
        self.refresh_config()
        self.set_model_name("default")
        return

    def set_model_name(self, model_name: str) -> None:
        self.__model_name = model_name

    def get_experiment_path(self) -> str:
        return p.join("..", "result", self.__experiment_id)

    def get_experiment_id(self) -> str:
        return self.__experiment_id

    def new_experiment(self):
        self.__experiment_id = "experiment" + str(time.time())
        print("experiment id: " + self.__experiment_id)
        RESULT_PATH = p.join("..", "result")
        os.makedirs(p.join(RESULT_PATH, self.__experiment_id))

    def refresh_config(self) -> None:
        old_library_list = self.__library_list
        config_path = p.join("..", "config", "config.yaml")
        f = open(config_path, "r", encoding="utf-8")
        config = yaml.full_load(f)
        f.close()
        self.__library_list = config["LIBRARY_LIST"].values()
        if old_library_list == self.__library_list:
            return
        self.__performer_list.clear()
        for library in self.__library_list:
            translator = translator_factory(library)
            self.__performer_list.append(translator)
            print(library + " translator loaded.")
        return

    def perform(self, abstract_model: dict, gen: int, index: int) -> list[dict]:
        result = []
        self.mo_co_assemble(abstract_model, gen, index, library="abstract")
        for performer in self.__performer_list:
            model_code = performer.translate(abstract_model)
            file_path = self.mo_co_assemble(model_code, gen, index, performer.get_library_name())
            model = performer.get_model_from_file(file_path)
            test_run_result, _1, _2, error_message = performer.run(model)
            test_run_result = True if test_run_result >= 0 else False
            if not test_run_result:
                train_result = False
                train_time_cost = -1
                run_time_cost = -1
                calculate_result = -1
                shape_result = []
            else:
                train_time_cost = performer.train(model)
                train_result = False if train_time_cost < 0 else True
                if not train_result:
                    run_time_cost = -1
                    calculate_result = -1
                    shape_result = []
                else:
                    run_time_cost, calculate_result, shape_result, _3 = performer.run(model)
            result_dict = {"run test": test_run_result, "train test": train_result,
                           "train time cost": train_time_cost, "run time cost": run_time_cost,
                           "calculate result": calculate_result, "case path": file_path,
                           "shape result": shape_result, "error message": error_message}
            result.append(result_dict)
        return result

    def mo_co_assemble(self, model_code: str | dict, gen: int, index: int, library: str) -> str:
        RESULT_PATH = p.join("..", "result", self.__experiment_id)
        case_id = self.__model_name + "-" + str(gen) + "-" + str(index)
        case_path = p.join(RESULT_PATH, case_id)
        if not p.exists(case_path):
            os.makedirs(case_path)
        if library != "abstract":
            file_path = p.join(case_path, case_id + "_" + library + ".py")
        else:
            file_path = p.join(case_path, case_id + "_abstract.yaml")
            f = open(file_path, "w", encoding="utf-8")
            yaml.dump(model_code, f)
            f.close()
            return file_path
        f = open(file_path, "w", encoding="utf-8")
        f.write(model_code)
        f.close()
        return file_path


concrete = Concrete()

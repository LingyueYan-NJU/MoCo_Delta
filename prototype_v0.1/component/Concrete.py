from _database import database
import os.path as p
import time

import yaml
import os


class Performer:
    def __init__(self):
        pass

    def get_library_name(self) -> str:
        pass

    def translate(self, abstract_model: dict) -> str:
        # Given a dict of abstract model, translate it into a str which can be written into a .py file
        # And then this .py file will be a whole model.
        pass

    def get_model_from_file(self, file_path: str):
        # Given a file path of a .py file, turn the net in this file to a model in memory.
        # Return this model.
        pass

    def train(self, model) -> float:
        # Given a model, train it, and return time cost(if failed, return -1).
        pass

    def run(self, model) -> (float, float):
        # Given a model, give a tensor and run it, then
        # 1. return time cost(-1 if failed).
        # 2. return the calculate result.
        pass


class TorchPerformer(Performer):
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "torch"

    def translate(self, abstract_model: dict) -> str:
        return "torch model code"

    def get_model_from_file(self, file_path: str):
        return "torch model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, float):
        return 1.0, 1.0


class PaddlePerformer(Performer):
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "paddle"

    def translate(self, abstract_model: dict) -> str:
        return "paddle model code"

    def get_model_from_file(self, file_path: str):
        return "paddle model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, float):
        return 1.0, 1.0


class MindSporePerformer(Performer):
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "mindspore"

    def translate(self, abstract_model: dict) -> str:
        return "mindspore model code"

    def get_model_from_file(self, file_path: str):
        return "mindspore model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, float):
        return 1.0, 1.0


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

    def run(self, model) -> (float, float):
        return 1.0, 1.0


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

    def run(self, model) -> (float, float):
        return 1.0, 1.0


def translator_factory(library: str) -> Performer:
    if library == "torch":
        return TorchPerformer()
    elif library == "paddle":
        return PaddlePerformer()
    elif library == "mindspore":
        return MindSporePerformer()
    elif library == "tensorflow":
        return TensorFlowPerformer()
    elif library == "jittor":
        return JittorPerformer()
    else:
        return TorchPerformer()


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
            test_run_result = True if performer.run(model)[0] > 0 else False
            if not test_run_result:
                train_result = False
                train_time_cost = -1
                run_time_cost = -1
                calculate_result = -1
            else:
                train_time_cost = performer.train(model)
                train_result = False if train_time_cost < 0 else True
                if not train_result:
                    run_time_cost = -1
                    calculate_result = -1
                else:
                    run_time_cost, calculate_result = performer.run(model)
            result_dict = {"test run result": test_run_result, "train result": train_result,
                           "train time cost": train_time_cost, "run time cost": run_time_cost,
                           "calculate result": calculate_result}
            result.append(result_dict)
        return result

    def mo_co_assemble(self, model_code: str | dict, gen: int, index: int, library: str) -> str:
        RESULT_PATH = p.join("..", "result", self.__experiment_id)
        case_id = self.__model_name + "-" + str(gen) + "-" + str(index)
        case_path = p.join(RESULT_PATH, case_id)
        if not p.exists(case_path):
            os.makedirs(case_path)
        if library != "abstract":
            file_path = p.join(case_path, library + "_version.py")
        else:
            file_path = p.join(case_path, library + "_version.yaml")
            f = open(file_path, "w", encoding="utf-8")
            yaml.dump(model_code, f)
            f.close()
            return file_path
        f = open(file_path, "w", encoding="utf-8")
        f.write(model_code)
        f.close()
        return file_path


seed = database.get_seed("lenet")
c = Concrete()

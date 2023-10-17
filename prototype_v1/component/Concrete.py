import os.path as p
import time
import yaml
import os
from MoCo_Delta.prototype_v1.component.Performer import Performer
from MoCo_Delta.prototype_v1.component.TorchPerformer import TorchPerformer
from MoCo_Delta.prototype_v1.component.PaddlePerformer import PaddlePerformer
from MoCo_Delta.prototype_v1.component.MindSporePerformer import MindSporePerformer
from MoCo_Delta.prototype_v1.component.TensorFlowPerformer import TensorFlowPerformer
from MoCo_Delta.prototype_v1.component.JittorPerformer import JittorPerformer


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


# seed = database.get_seed("lenet")
# c = Concrete()
concrete = Concrete()

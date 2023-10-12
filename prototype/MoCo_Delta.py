import copy
import os
import random

import yaml
from database_manager import DBManager
import mutate
from exec import Exec_pytorch, Exec_paddle


class MoCo_Delta:
    def __init__(self, model_name="LeNet"):
        self.model_name = model_name
        self.model_name = "LeNet"
        self.n = 2
        self.dbm = DBManager()
        self.seed_model: dict = self.load_seed_model(self.model_name)
        self.m = mutate.Mutator()
        self.exec_list = [Exec_pytorch(), Exec_paddle()]
        self.case_path = os.path.join(".", "case")

    def load_seed_model(self, model_name: str) -> dict:
        return self.dbm.get_seed_model(model_name)

    def mutate(self, layer: dict) -> dict:
        line = self.dict_to_line(layer)
        new_line = self.m.api_mutate(line)[0]
        return {"layer": mutate.get_function(new_line), "params": mutate.get_params(new_line)}

    def dict_to_line(self, layer_and_para_dict: dict) -> str:
        return mutate.generate_line(layer_and_para_dict["layer"], layer_and_para_dict["params"])

    def test_model(self, abs_model: dict, case_name: str) -> bool:
        path = os.path.join(self.case_path, case_name)
        if not os.path.exists(path):
            os.makedirs(path)
        f = open(os.path.join(path, "abs_model.yaml"), "w")
        yaml.dump(abs_model, f)
        f.close()
        result_list = []
        for Exec in self.exec_list:
            result_list.append(Exec.get_result(model=abs_model, case_path=path))
        flag = True
        reason = ["run", "train", "shape", "shape", "data"]
        for i in range(len(result_list[0])):
            if result_list[0][i] - result_list[1][i] > 0.85:
                flag = False
                f = open(os.path.join(path, "log.txt"), "a", encoding="utf-8")
                f.write("fail\n")
                f.write(reason[i])
                f.write("\n\n\n")
                f.close()
        return flag

    def assemble_code_tree(self):
        seed = self.seed_model
        if not os.path.exists(self.case_path):
            os.makedirs(self.case_path)
        count_pass = 0
        count_err = 0
        count = 0
        template: dict = {"input": seed["input"], "output": seed["output"], "hidden_layer": []}
        queue = [template]
        layer = 0
        for ele in seed["hidden_layer"]:
            this_layer_count = 0
            this_layer_pass = 0
            this_layer_err = 0
            if len(queue) == 0:
                break
            if isinstance(ele, str):
                for model in queue:
                    model["hidden_layer"].append(ele)
            elif isinstance(ele, dict):
                pass_list = []
                for model in queue:
                    for i in range(self.n):
                        temp = copy.deepcopy(model)
                        temp["hidden_layer"].append(self.mutate(ele))
                        case_name = self.model_name + "-" + str(layer + 1) + "-" + str(this_layer_count + 1)
                        this_layer_count += 1
                        if self.test_model(temp, case_name):
                            pass_list.append(temp)
                            count_pass += 1
                            this_layer_pass += 1
                            print(case_name + " 组装完成, 成功")
                        else:
                            count_err += 1
                            this_layer_err += 1
                            print(case_name + " 组装完成, 失败")
                queue = pass_list
                layer += 1
            print(str(layer) + "层: 成功" + str(this_layer_pass) + "个, 失败" + str(this_layer_err) + "个。")
        print("总: 成功" + str(count_pass) + "个, 失败" + str(count_err) + "个。")


mcd = MoCo_Delta()

import os
from _database import database
from Concrete import concrete
from Mutate import mutator
from ResultAnalyse import analyser
import os.path as p
import yaml
import copy


config = {"N": 3, "LIBRARY_LIST": []}


def set_config():
    CONFIG_PATH = p.join("..", "config", "config.yaml")
    f = open(CONFIG_PATH, "r", encoding="utf-8")
    file_config = yaml.full_load(f)
    f.close()
    config["LIBRARY_LIST"].clear()
    for library in list(file_config["LIBRARY_LIST"].values()):
        config["LIBRARY_LIST"].append(library)
    config["N"] = file_config["N"]


def refresh__config():
    set_config()
    mutator.refresh_config()
    concrete.new_experiment()
    database.refresh_config()
    concrete.refresh_config()
    analyser.refresh_config()


def check() -> bool:
    MAIN_PATH = ".."
    REPORT_PATH = p.join(MAIN_PATH, "report")
    RESULT_PATH = p.join(MAIN_PATH, "result")
    if not p.exists(REPORT_PATH):
        os.makedirs(REPORT_PATH)
    if not p.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    return True


def go(net: str = "LeNet") -> None:
    check()
    set_config()
    seed = database.get_seed(net)
    concrete.set_model_name(net)
    report_f = open(p.join(concrete.get_experiment_path(), "main_report.txt"), "w", encoding="utf-8")
    report_f.write("id: " + concrete.get_experiment_id() + "\npath: " + concrete.get_experiment_path() + "\n\n")
    report_f.close()
    report_f = open(p.join(concrete.get_experiment_path(), "main_report.txt"), "a", encoding="utf-8")
    count_pass = 0
    count_err = 0
    count = 0
    template: dict = {net: []}
    for model in seed.keys():
        if model != net:
            template[model] = seed[model]
    queue = [template]
    layer = 0
    for ele in seed[net]:
        this_layer_count = 0
        this_layer_pass = 0
        this_layer_err = 0
        if len(queue) == 0:
            break
        if ele["layer"] == "cat":
            for model in queue:
                model[net].append(ele)
                # just for cat now
        elif isinstance(ele, dict):
            pass_list = []
            for model in queue:
                for i in range(config["N"]):
                    temp = copy.deepcopy(model)
                    mutated_dict, mutate_info = mutator.mutate(ele)
                    temp[net].append(mutated_dict)
                    gen = layer + 1
                    index = this_layer_count + 1
                    this_layer_count += 1
                    result = concrete.perform(temp, gen, index)
                    for r in result:
                        r["mutate info"] = mutate_info
                    ana = analyser.analyse_result(result)
                    if ana:
                        pass_list.append(temp)
                        count_pass += 1
                        this_layer_pass += 1
                        print(net + "-" + str(gen) + "-" + str(index) + " 组装完成, 成功")
                    else:
                        count_err += 1
                        this_layer_err += 1
                        print(net + "-" + str(gen) + "-" + str(index) + " 组装完成, 失败")
            queue = pass_list
            layer += 1
        report = str(layer) + "层: 成功" + str(this_layer_pass) + "个, 失败" + str(this_layer_err) + "个。\n"
        report_f.write(report)
        print(report)
    report = "总: 成功" + str(count_pass) + "个, 失败" + str(count_err) + "个。\n"
    report_f.write(report)
    print(report)
    report_f.close()


# if __name__ == "__main__":
#     go("LeNet")

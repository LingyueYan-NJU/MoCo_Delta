import os
import yaml
import copy

DENSENET_PATH = "./densenet.py"
SQUEEZENET_PATH = "./squeezenet.py"

f = open(SQUEEZENET_PATH, "r")
lines = f.readlines()
f.close()


def get_params(line: str) -> dict:  # 注: 解析单句的参数列表, 例如torch.nn.Conv2d(...), 必须带上参数名
    ana = line.split('(', 1)[1][:-1]
    ana = ana.replace(' ', '')
    temp_str = ''
    params_dict = {}
    index = 0
    while index < len(ana):
        i = index
        while i < len(ana) and ana[i] != '=':
            temp_str = temp_str + ana[i]
            i = i + 1
        name = copy.deepcopy(temp_str)
        i = i + 1
        temp_str = ''
        while i < len(ana) and not (ana[i] == ',' and not (ana[i + 1].isdigit())):
            temp_str = temp_str + ana[i]
            i = i + 1
        value = copy.deepcopy(temp_str)
        temp_str = ''
        if i < len(ana):
            i = i + 1
        params_dict[name] = value
        index = i
    return params_dict


def get_function(line: str) -> str:  # 注: 解析单句的函数名, 如torch.nn.Conv2d(...)
    return line.split('(', 1)[0].split("=", 1)[1]


define_lines = lines[8:56]
define_to_function_dict = {}
for line in define_lines:
    pure_line = line.replace(" ", "").replace("\n", "").replace("self.", "").replace("torch.nn.", "")
    define = pure_line.split("=", 1)[0]
    param_dict = get_params(pure_line)
    layer_name = get_function(pure_line).lower()
    current_dict = {"layer": layer_name, "params": param_dict, "in": "x", "out": "x"}
    define_to_function_dict[define] = current_dict


forward_lines = lines[58:122]
result_list = []
for line in forward_lines:
    pure_line = line.replace(" ", "").replace("\n", "").replace("torch.nn.", "")
    if "self." in line:
        out = pure_line.split("=", 1)[0]
        _in = pure_line.split("(", 1)[1].split(")", 1)[0]
        define = pure_line.split("self.", 1)[1].split("(", 1)[0]
        result_dict = copy.deepcopy(define_to_function_dict[define])
        result_dict["in"] = _in
        result_dict["out"] = out
        result_list.append(result_dict)
    elif "torch.cat" in line:
        result_dict = {"layer": "cat", "params": {"dims": "-1"}, "in": "?", "out": "?"}
        _in = "[" + pure_line.split("[", 1)[1].split("]", 1)[0] + "]"
        out = pure_line.split("=", 1)[0]
        dims = pure_line.split("dim=", 1)[1].split(")", 1)[0]
        result_dict["params"]["dims"] = dims
        result_dict["in"] = _in
        result_dict["out"] = out
        result_list.append(result_dict)

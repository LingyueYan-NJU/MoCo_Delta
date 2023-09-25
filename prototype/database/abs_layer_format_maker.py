import os
import yaml


TORCH_LAYER_INFO_PATH = os.path.join(".", "torch_layer_info")
PADDLE_LAYER_INFO_PATH = os.path.join(".", "paddle_layer_info")
ABS_LAYER_FORMAT_PATH = os.path.join(".", "Abs_Layer_Format")


def get_para_name_list(api_name: str, lib: str) -> list:
    if lib == "paddle":
        path = PADDLE_LAYER_INFO_PATH
    else:
        path = TORCH_LAYER_INFO_PATH
    api = api_name
    f = open(os.path.join(path, api + ".yaml"))
    layer_info: dict = yaml.full_load(f)
    return list(layer_info["constraints"].keys())


torch_layer_name_list = []
temp = os.listdir(TORCH_LAYER_INFO_PATH)
for layer_name in temp:
    torch_layer_name_list.append(layer_name[:-5])
paddle_layer_name_list = []
temp = os.listdir(PADDLE_LAYER_INFO_PATH)
for layer_name in temp:
    paddle_layer_name_list.append(layer_name[:-5])
torch_layer_and_its_params = {}
paddle_layer_and_its_params = {}
for torch_layer_name in torch_layer_name_list:
    torch_layer_and_its_params[torch_layer_name.lower()[9:]] = get_para_name_list(torch_layer_name, "torch"), \
        torch_layer_name
for paddle_layer_name in paddle_layer_name_list:
    paddle_layer_and_its_params[paddle_layer_name.lower()[10:]] = get_para_name_list(paddle_layer_name, "paddle"), \
        paddle_layer_name
for torch_layer_name_lower in torch_layer_and_its_params.keys():
    if torch_layer_name_lower != "finished!":
        continue
    d = {}
    name = torch_layer_name_lower.replace("torch.nn.", "")
    torch_param_list = torch_layer_and_its_params[torch_layer_name_lower][0]
    torch_true_name = torch_layer_and_its_params[torch_layer_name_lower][1]
    api = {"torch": torch_true_name, "paddle": "to be decided"}
    params = {}
    if name in paddle_layer_and_its_params.keys():
        api["paddle"] = paddle_layer_and_its_params[name][1]
        paddle_param_list = paddle_layer_and_its_params[name][0]
        for para in torch_param_list:
            params[para] = {"torch": para, "paddle": "to be decided"}
            if para in paddle_param_list:
                params[para]["paddle"] = para
    else:
        for para in torch_param_list:
            params[para] = {"torch": para, "paddle": "to be decided"}
    d["api"] = api
    d["params"] = params
    f = open(os.path.join(ABS_LAYER_FORMAT_PATH, name + ".yaml"), "w", encoding="utf-8")
    yaml.dump(d, f)
    f.close()




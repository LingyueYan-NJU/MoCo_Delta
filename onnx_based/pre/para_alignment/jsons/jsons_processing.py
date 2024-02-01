import copy
import os
import json


parameter_info_path = "./original/parameter_infos.json"
parameter_mapping_path = "./original/parameter_mapping.json"
op_types_mapping_path = "./original/op_types_mapping.json"

f = open(parameter_info_path, "r", encoding="utf-8")
parameter_info = json.load(f)
f.close()
f = open(parameter_mapping_path, "r", encoding="utf-8")
parameter_mapping = json.load(f)
f.close()
f = open(op_types_mapping_path, "r", encoding="utf-8")
op_types_mapping = json.load(f)
f.close()
onnx_apis = list(parameter_info.keys())

# # parameter info processing (all false)
# for onnx_api in onnx_apis:
#     tbm = copy.deepcopy(parameter_info[onnx_api]["pytorch"])
#     for torch_api in tbm.keys():
#         tbm2 = copy.deepcopy(tbm[torch_api])
#         for para_name in tbm2.keys():
#             tbm2[para_name]["isRequired"] = False
#         tbm[torch_api] = tbm2
#     parameter_info[onnx_api]["pytorch"] = tbm
#
# f = open("./new_parameter_info.json", "w", encoding="utf-8")
# json.dump(parameter_info, f, indent=4)
# f.close()

# # parameter mapping processing (all tbd)
# for onnx_api in parameter_mapping.keys():
#     current_onnx_api = onnx_api
#     dict_to_make = {}
#     torch_apis_to_make = op_types_mapping[current_onnx_api]["pytorch"]["apis"]
#     onnx_paras_to_make = os.listdir(f"../../para_info/{current_onnx_api}/onnx/attributes")
#     for torch_api in torch_apis_to_make:
#         dict_to_make[torch_api] ={}
#         for onnx_para in onnx_paras_to_make:
#             dict_to_make[torch_api][onnx_para[:-4]] = "tbdtbdtbdtbdtbdtbdtbd"
#     parameter_mapping[current_onnx_api]["pytorch"] = dict_to_make
#
# f = open("./new_parameter_mapping.json", "w", encoding="utf-8")
# json.dump(parameter_mapping, f, indent=4)
# f.close()

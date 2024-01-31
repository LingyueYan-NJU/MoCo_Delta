import json


def write_in(d: dict, onnx_api_name):
    f = open(f"./ana_result/{onnx_api_name}/{onnx_api_name}.json", "w", encoding="utf-8")
    f2 = open(f"./ana_result/{onnx_api_name}/{onnx_api_name}.txt", "w", encoding="utf-8")
    json.dump(d, f)
    json.dump(d, f2)
    f.close()
    f2.close()
    return

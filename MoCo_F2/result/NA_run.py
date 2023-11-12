import sys
import traceback
import torch
from importlib import import_module
import os


def run_na(net: str):
    f = open("./NA_report_" + net + ".txt", "a", encoding="utf-8")
    model_name = "MoCoNA"
    file_list = os.listdir(".")
    target_path = ""
    for file in file_list:
        if net in file and (not file.endswith("txt")):
            target_path = os.path.join(".", file)
            break
        else:
            continue
    if target_path == "":
        print("no such dir: " + net)
        return
    file_list = os.listdir(target_path)
    for file in file_list:
        if net in file:
            current_case = file
            pth = os.path.join(target_path, file)
            if "MoCoNA.py" not in os.listdir(pth):
                continue
            sys.path.append(pth)
            try:
                import_module(model_name)
                result_report = "OK"
            except Exception:
                result_report = ""
                result_report += "From " + current_case + "\n"
                result_report += "code: "
                with open(os.path.join(pth, "MoCoNA.py"), "r", encoding="utf-8") as ff:
                    result_report += ff.read()
                result_report += "\nerror message: " + str(traceback.format_exc()) + "\n"
                result_report += "\n\n===================================\n\n"
            if result_report != "OK":
                f.write(result_report)
            sys.path.remove(pth)
            print(current_case + " scan finished.")
    f.close()
    return


for net in ["LeNet", "alexnet", "googlenet"]:
    run_na(net)

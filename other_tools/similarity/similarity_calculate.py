import os.path

import yaml


def read_single_directory(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    file_list = os.listdir(path)
    result = {}
    for file_name in file_list:
        if file_name.endswith("yaml"):
            file_path = os.path.join(path, file_name)
            api_name = file_name[:-5]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    d = yaml.full_load(f)
                    f.close()
                result[api_name] = d
            except Exception:
                print(api_name + " cannot read.")
    return result


class DBManager:
    def __init__(self):
        self.information_path = os.path.join(".", "..", "..", "information")
        self.information = {}
        lib_list = ["jittor", "torch", "paddle", "ms"]
        for lib in lib_list:
            path = os.path.join(self.information_path, lib + "_layer_info")
            self.information[lib] = read_single_directory(path)
        return


db = DBManager()

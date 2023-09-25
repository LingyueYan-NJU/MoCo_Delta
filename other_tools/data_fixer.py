import os
import yaml
path = os.path.join(".", "..", "information", "paddle_layer_info")
file_list = os.listdir(path)
for file_name in file_list:
    file_path = os.path.join(path, file_name)
    f = open(file_path, "r", encoding="utf-8")
    info = yaml.full_load(f)
    f.close()
    with open(file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(info, yaml_file, allow_unicode=True, sort_keys=False)


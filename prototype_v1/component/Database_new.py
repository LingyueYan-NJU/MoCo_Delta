import copy
import os
import os.path as p
import yaml
import csv

database_path = p.join("..", "database")
implicit_database_path = p.join(database_path, "implicit")
config_path = p.join("..", "config", "config.yaml")


class Database:
    def __init__(self):
        self.__library_list = []
        self.__threshold = 0.0
        self.__mode = 0
        self.__read_config()
        self.__implicit_layer_info = {}
        self.__implicit_layer_similarity = {}
        self.__implicit_layer_similarity_valid = {}
        self.__api_name_map = {}
        self.__api_name_map_valid = {}
        self.__inverse_api_name_map = {}
        self.__inverse_api_name_map_valid = {}
        self.__all_api_list = {}
        self.__candidate_map = {}
        self.__api_para_map = {}
        self.__abstract_api_layer_info = {}
        self.__total_refresh()

    def refresh_config(self):
        pass

    def __total_refresh(self):
        self.__read_implicit_layer_info()
        self.__read_implicit_layer_similarity()
        self.__calculate_implicit_layer_similarity_valid()
        self.__read_api_name_map_and_inverse_api_name_map()
        self.__calculate_api_name_map_valid()
        self.__calculate_inverse_api_name_map_valid()
        self.__calculate_all_api_list()
        self.__calculate_candidate_map()
        self.__read_api_para_map()
        self.__calculate_abstract_api_layer_info()

    def __part_refresh_for_threshold(self):
        self.__calculate_implicit_layer_similarity_valid()
        self.__calculate_candidate_map()

    def __read_config(self):
        # unit test passed
        print("### initializing config")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.full_load(f)
        # check
        assert (config["MODE"] == 1 and len(config["LIBRARY_LIST"].values()) == 1) or \
               (config["MODE"] == 2 and len(config["LIBRARY_LIST"].values()) >= 2), "MODE and LIST are not matched"
        library_dict = config["LIBRARY_LIST"]
        self.__library_list = list(library_dict.values())
        self.__threshold = config["THRESHOLD"]
        self.__mode = config["MODE"]
        return

    def __read_implicit_layer_info(self):
        # unit test passed
        print("### initializing layer info")
        self.__implicit_layer_info = {}
        dir_list = os.listdir(implicit_database_path)
        for library in self.__library_list:
            assert library in dir_list, "check config"
        for library in self.__library_list:
            current_library_layer_info = {}
            current_layer_info_path = p.join(implicit_database_path, library, "layer_info")
            file_list = os.listdir(current_layer_info_path)
            for file_name in file_list:
                current_file_path = p.join(current_layer_info_path, file_name)
                with open(current_file_path, "r", encoding="utf-8") as f:
                    current_info = yaml.full_load(f)
                    current_layer_name = file_name[:-5]
                    current_library_layer_info[current_layer_name] = current_info
            self.__implicit_layer_info[library] = current_library_layer_info
        return

    def __read_implicit_layer_similarity(self):
        # unit test passed
        print("### initializing similarity")
        for library in self.__library_list:
            current_similarity_file = p.join(implicit_database_path, library, "layer_similarity.yaml")
            with open(current_similarity_file, "r", encoding="utf-8") as f:
                self.__implicit_layer_similarity[library] = yaml.full_load(f)
        return

    def __calculate_implicit_layer_similarity_valid(self):
        # unit test passed
        for library in self.__library_list:
            old_dict = self.__implicit_layer_similarity[library]
            new_dict = {}
            for api in old_dict.keys():
                current_dict = old_dict[api]
                new_current_dict = {}
                for simi_api in current_dict:
                    if simi_api == api:
                        continue
                    elif current_dict[simi_api] < self.__threshold:
                        continue
                    else:
                        new_current_dict[simi_api] = current_dict[simi_api]
                new_dict[api] = new_current_dict
            self.__implicit_layer_similarity_valid[library] = new_dict
        return

    def __read_api_name_map_and_inverse_api_name_map(self):
        # unit test passed
        print("### initializing maps")
        target_csv_path = p.join(database_path, "abstract", "abstract_api_name.csv")
        f = open(target_csv_path, "r")
        reader = csv.reader(f)
        row_list = []
        for row in reader:
            if len(row) > 2:
                row_list.append(row)
        head = row_list[0]
        row_list = row_list[1:]
        valid_locations = []
        for library in self.__library_list:
            valid_locations.append(head.index(library))
        self.__api_name_map = {}
        self.__inverse_api_name_map = {}
        for library in self.__library_list:
            self.__inverse_api_name_map[library] = {}
        for row in row_list:
            api_id = row[0]
            abstract_api_name = row[1]
            implicit_library_api_name = []
            for location in valid_locations:
                implicit_library_api_name.append(row[location])
            dict_for_main_map = {"id": api_id}
            for i in range(len(self.__library_list)):
                dict_for_main_map[self.__library_list[i]] = implicit_library_api_name[i] if \
                    len(implicit_library_api_name[i]) > 5 else "NOAPI"
            self.__api_name_map[abstract_api_name] = dict_for_main_map
            for i in range(len(self.__library_list)):
                self.__inverse_api_name_map[self.__library_list[i]][implicit_library_api_name[i]] = abstract_api_name
            for i in range(len(self.__library_list)):
                if "" in self.__inverse_api_name_map[self.__library_list[i]].keys():
                    self.__inverse_api_name_map[self.__library_list[i]].pop("")
                if "NOAPI" in self.__inverse_api_name_map[self.__library_list[i]].keys():
                    self.__inverse_api_name_map[self.__library_list[i]].pop("NOAPI")
        return

    def __calculate_api_name_map_valid(self):
        # unit test passed
        for abstract_api_name in self.__api_name_map:
            implicit_apis = self.__api_name_map[abstract_api_name].values()
            if "" not in implicit_apis and "NOAPI" not in implicit_apis:
                self.__api_name_map_valid[abstract_api_name] = self.__api_name_map[abstract_api_name]
        return

    def __calculate_inverse_api_name_map_valid(self):
        # unit test passed
        valid_list = list(self.__api_name_map_valid.keys())
        for library in self.__library_list:
            old_dict = self.__inverse_api_name_map[library]
            new_dict = {}
            for implicit_api_name in old_dict.keys():
                if old_dict[implicit_api_name] not in valid_list:
                    continue
                else:
                    new_dict[implicit_api_name] = old_dict[implicit_api_name]
            self.__inverse_api_name_map_valid[library] = new_dict
        return

    def __calculate_all_api_list(self):
        # unit test passed
        self.__all_api_list = {"abstract": list(self.__api_name_map_valid.keys())}
        for library in self.__library_list:
            self.__all_api_list[library] = self.__inverse_api_name_map_valid[library].keys()
        return

    def __calculate_candidate_map(self):
        pass

    def __read_api_para_map(self):
        # TODO
        pass

    def __calculate_abstract_api_layer_info(self):
        # TODO
        pass


d = Database()

import copy
import os
import os.path as p
import yaml
import csv


def merge(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2
    intersection_list = list(intersection)
    return intersection_list


class Database:
    def __init__(self):
        self.__database_path = p.join("..", "database")
        self.__implicit_database_path = p.join(self.__database_path, "implicit")

        print("### Database initializing...")

        # read config
        print("### initializing config")
        __config_path = p.join("..", "config", "config.yaml")
        with open(__config_path, "r", encoding="utf-8") as f:
            config = yaml.full_load(f)
        # check
        assert (config["MODE"] == 1 and len(config["LIBRARY_LIST"].values()) == 1) or \
               (config["MODE"] == 2 and len(config["LIBRARY_LIST"].values()) >= 2), "MODE and LIST are not matched"
        self.__library_dict = config["LIBRARY_LIST"]
        self.__library_list = list(self.__library_dict.values())
        THRESHOLD = config["THRESHOLD"]
        self.mode = config["MODE"]
        # config read over

        # read implicit layer info
        print("### initializing layer info")
        self.__implicit_layer_info = {}
        dir_list = os.listdir(self.__implicit_database_path)
        for library in self.__library_list:
            assert library in dir_list, "check config"
        for library in self.__library_list:
            current_library_layer_info = {}
            current_layer_info_path = p.join(self.__implicit_database_path, library, "layer_info")
            file_list = os.listdir(current_layer_info_path)
            for file_name in file_list:
                current_file_path = p.join(current_layer_info_path, file_name)
                with open(current_file_path, "r", encoding="utf-8") as f:
                    current_info = yaml.full_load(f)
                    current_layer_name = file_name[:-5]
                    current_library_layer_info[current_layer_name] = current_info
            self.__implicit_layer_info[library] = current_library_layer_info
        # implicit layer info read over

        # read implicit layer similarity
        print("### initializing similarity")
        self.__layer_similarity = {}
        for library in self.__library_list:
            current_similarity_file = p.join(self.__implicit_database_path, library, "layer_similarity.yaml")
            with open(current_similarity_file, "r", encoding="utf-8") as f:
                self.__layer_similarity[library] = yaml.full_load(f)
        # implicit layer similarity read over

        # read abstract-to-implicit api_name table
        print("### initializing maps")
        target_csv_path = p.join(self.__database_path, "abstract", "abstract_api_name.csv")
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
        self.__main_api_name_map = {}
        self.__inverse_map = {}
        for library in self.__library_list:
            self.__inverse_map[library] = {}
        for row in row_list:
            api_id = row[0]
            abstract_api_name = row[1]
            implicit_library_api_name = []
            for location in valid_locations:
                implicit_library_api_name.append(row[location])
            dict_for_main_map = {"id": api_id}
            for i in range(len(self.__library_list)):
                dict_for_main_map[self.__library_dict[i]] = implicit_library_api_name[i] if \
                    len(implicit_library_api_name[i]) > 5 else "NOAPI"
            self.__main_api_name_map[abstract_api_name] = dict_for_main_map
            for i in range(len(self.__library_list)):
                self.__inverse_map[self.__library_list[i]][implicit_library_api_name[i]] = abstract_api_name
            for i in range(len(self.__library_list)):
                if "" in self.__inverse_map[self.__library_list[i]].keys():
                    self.__inverse_map[self.__library_list[i]].pop("")
                if "NOAPI" in self.__inverse_map[self.__library_list[i]].keys():
                    self.__inverse_map[self.__library_list[i]].pop("NOAPI")
        self.__main_api_para_map = {}
        if self.mode == 1:
            for abstract_api_name in self.__main_api_name_map.keys():
                self.__main_api_para_map[abstract_api_name] = {}
                implicit_api_name = self.get_implicit_api_name(self.__library_list[0], abstract_api_name)
                implicit_layer_info = self.get_implicit_layer_info(self.__library_list[0], implicit_api_name)
                paras = list(implicit_layer_info["constraints"].keys())
                for para in paras:
                    abstract_para_name = para
                    self.__main_api_para_map[abstract_api_name][abstract_para_name] = {}
                    self.__main_api_para_map[abstract_api_name][abstract_para_name][self.__library_list[0]] = para
        else:
            self.__main_api_para_map = {}
            # TODO
            pass
        # abstract-to-implicit api_name table read over

        self.__threshold = THRESHOLD

        # initializing candidate map
        print("### initializing candidate map")
        self.__candidate_map = self.__just_get_candidate_dict(self.__threshold)
        # candidate map initialize over

        # initialize(calculate) abstract layer info
        print("### initializing abstract layer info")
        # calculate abstract layer info here
        self.__abstract_layer_info = {}
        if self.mode == 1:
            for abstract_api_name in list(self.__main_api_name_map.keys()):
                self.__abstract_layer_info[abstract_api_name] = self.get_implicit_layer_info(
                    self.__library_list[0], self.get_implicit_api_name(self.__library_list[0], abstract_api_name))
        else:
            self.__abstract_layer_info = {}
            # TODO
            pass
        # abstract layer info initialize over

        # initialize an abstract similarity dict, this is just for mode 1.

        print("### Database OK")
        return

    def is_library_valid(self, library: str) -> bool:
        return library in self.__library_list

    def is_abstract_api_name_valid(self, abstract_api_name: str) -> bool:
        return abstract_api_name in list(self.__main_api_name_map.keys())

    def is_implicit_api_name_valid(self, library: str, implicit_api_name: str) -> bool:
        return self.is_library_valid(library) and (implicit_api_name in list(self.__inverse_map[library].keys()))

    def set_threshold(self, threshold: float) -> None:
        assert 0.01 <= threshold <= 0.99, "check threshold, between 0.01 and 0.99."
        self.__threshold = threshold
        self.__candidate_map = self.__just_get_candidate_dict(self.__threshold)
        print("Threshold changed to " + str(threshold) + ", map updated.")
        return

    def get_threshold(self) -> float:
        return self.__threshold

    def __get_all_api_list(self, library: str) -> list:
        assert self.is_library_valid(library), "check library name"
        return list(self.__implicit_layer_info[library].keys())

    def __get_api_description(self, library: str, implicit_api_name: str) -> str:
        assert self.is_implicit_api_name_valid(library, implicit_api_name), "check input"
        flag = False
        for lib in self.__library_list:
            if implicit_api_name.startswith(lib):
                pass
            else:
                flag = True
        if not flag:
            return self.__implicit_layer_info[library][library + ".nn." + implicit_api_name]["descp"]
        else:
            return self.__implicit_layer_info[library][implicit_api_name]["descp"]

    def get_api_similarity(self, library: str, implicit_api_name: str) -> dict:
        assert self.is_implicit_api_name_valid(library, implicit_api_name), "check input"
        return self.__layer_similarity[library][implicit_api_name]

    def get_implicit_api_name(self, library: str, abstract_api_name: str) -> str:
        assert self.is_library_valid(library), "check input library"
        assert self.is_abstract_api_name_valid(abstract_api_name), "check input abstract_api_name"
        return self.__main_api_name_map[abstract_api_name][library]

    def get_abstract_api_name(self, library: str, implicit_api_name: str) -> str:
        assert self.is_implicit_api_name_valid(library, implicit_api_name), "check input"
        return self.__inverse_map[library][implicit_api_name]

    def get_candidate_mutate_list(self, abstract_api_name: str) -> list[str]:
        assert self.is_abstract_api_name_valid(abstract_api_name)
        return self.__candidate_map[abstract_api_name]

    def __get_candidate_mutate_list(self, abstract_api_name: str, threshold: float) -> list[str]:
        assert abstract_api_name in list(self.__main_api_name_map.keys()), "check input abstract_api_name"
        abstract_candidate_api_list = list(self.__main_api_name_map.keys())
        for library in self.__library_list:
            current_implicit_api_name = self.get_implicit_api_name(library, abstract_api_name)
            if current_implicit_api_name == "NOAPI" or current_implicit_api_name == "":
                continue
            implicit_candidate_api_list = []
            current_abstract_candidate_api_list = []
            similarity_dict = self.get_api_similarity(library, current_implicit_api_name)
            for layer in similarity_dict.keys():
                if similarity_dict[layer] >= threshold and layer in self.__inverse_map[library].keys():
                    implicit_candidate_api_list.append(layer)
            for layer in implicit_candidate_api_list:
                current_abstract_candidate_api_list.append(self.get_abstract_api_name(library, layer))
            abstract_candidate_api_list = merge(abstract_candidate_api_list, current_abstract_candidate_api_list)
        if abstract_api_name in abstract_candidate_api_list:
            abstract_candidate_api_list.remove(abstract_api_name)
        return abstract_candidate_api_list

    def __get_candidate_dict(self, threshold: float) -> (dict, int):
        zero_num = 0
        # abstract_api_num = len(self.main_api_name_map)
        result_dict = {}
        abstract_api_list = list(self.__main_api_name_map.keys())
        for abstract_api_name in abstract_api_list:
            candidate_list = self.__get_candidate_mutate_list(abstract_api_name, threshold)
            if len(candidate_list) == 0:
                zero_num += 1
            result_dict[abstract_api_name] = candidate_list
        return result_dict, zero_num

    def __just_get_candidate_dict(self, threshold: float) -> dict:
        return self.__get_candidate_dict(threshold)[0]

    def __generate_candidate_report(self, threshold: float) -> (str, int):
        candidate_dict, zero_num = self.__get_candidate_dict(threshold)
        report = "*With the threshold of " + str(threshold) + ", there are " + str(zero_num) + \
                 " apis that cannot be mutated\n\n"
        for abstract_api_name in candidate_dict.keys():
            report += (abstract_api_name + " :\n")
            report += str(candidate_dict[abstract_api_name])
            report += "\n"
        return report, zero_num

    def generate_candidate_report(self, threshold: float = None) -> None:
        if threshold is not None:
            report = self.__generate_candidate_report(threshold=threshold)[0]
        else:
            min_threshold = 0.0
            report = ""
            for th in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                current_report, zero_num = self.__generate_candidate_report(th)
                report += current_report
                if zero_num == 0:
                    min_threshold = th
                report += "\n\n\n"
            recommend = "While with the threshold of " + str(min_threshold) + ", all api can be mutated.\n"
            report = recommend + report
        report_path = p.join("..", "report", "apiCandidateReport.txt")
        f = open(report_path, "w", encoding="utf-8")
        f.write(report)
        f.close()
        return

    def get_implicit_para_name(self, library: str, abstract_api_name: str, abstract_para_name: str) -> str:
        assert self.is_library_valid(library), "check library input"
        assert self.is_abstract_api_name_valid(abstract_api_name), "check abstract_api_name input"
        if self.mode == 1:
            return self.__main_api_para_map[abstract_api_name][abstract_para_name][library]
        elif self.mode == 2:
            # TODO
            return "no para"

    def get_implicit_layer_info(self, library: str, implicit_api_name: str) -> dict:
        assert self.is_library_valid(library), "check library input"
        assert self.is_implicit_api_name_valid(library, implicit_api_name), "check implicit_api_name input"
        return self.__implicit_layer_info[library][implicit_api_name]

    def get_abstract_layer_info(self, abstract_api_name: str) -> dict:
        assert self.is_abstract_api_name_valid(abstract_api_name), "check abstract_api_name input"
        if self.mode == 1:
            return self.__abstract_layer_info[abstract_api_name]
        elif self.mode == 2:
            # TODO
            return {}
        else:
            return {}

    def get_seed(self, seed_name: str) -> dict:
        SEED_PATH = p.join(self.__database_path, "seed", seed_name + ".yaml")
        f = open(SEED_PATH, "r")
        seed = yaml.full_load(f)
        f.close()
        return seed

    def refresh(self) -> None:
        self.__init__()
        return

    def refresh_config(self) -> None:
        old_list = copy.deepcopy(self.__library_list)
        print("### initializing config")
        __config_path = p.join("..", "config", "config.yaml")
        with open(__config_path, "r", encoding="utf-8") as f:
            config = yaml.full_load(f)
        self.__library_dict = config["LIBRARY_LIST"]
        self.__library_list = list(self.__library_dict.values())
        if old_list == self.__library_list:
            self.set_threshold(config["THRESHOLD"])
            return
        else:
            self.refresh()
            self.set_threshold(config["THRESHOLD"])
            return


d = Database()
# d.generate_candidate_report(None)

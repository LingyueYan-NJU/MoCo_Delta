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
        self.database_path = p.join("..", "database")
        self.implicit_database_path = p.join(self.database_path, "implicit")

        print("### Database initializing...")

        # read config
        print("### initializing config")
        config_path = p.join("..", "config", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.full_load(f)
        self.library_dict = config["LIBRARY_LIST"]
        self.library_list = list(self.library_dict.values())
        # config read over

        # read implicit layer info
        print("### initializing layer info")
        self.layer_info = {}
        dir_list = os.listdir(self.implicit_database_path)
        for library in self.library_list:
            assert library in dir_list, "check config"
        for library in self.library_list:
            current_library_layer_info = {}
            current_layer_info_path = p.join(self.implicit_database_path, library, "layer_info")
            file_list = os.listdir(current_layer_info_path)
            for file_name in file_list:
                current_file_path = p.join(current_layer_info_path, file_name)
                with open(current_file_path, "r", encoding="utf-8") as f:
                    current_info = yaml.full_load(f)
                    current_layer_name = file_name[:-5]
                    current_library_layer_info[current_layer_name] = current_info
            self.layer_info[library] = current_library_layer_info
        # implicit layer info read over

        # read implicit layer similarity
        print("### initializing similarity")
        self.layer_similarity = {}
        for library in self.library_list:
            current_similarity_file = p.join(self.implicit_database_path, library, "layer_similarity.yaml")
            with open(current_similarity_file, "r", encoding="utf-8") as f:
                self.layer_similarity[library] = yaml.full_load(f)
        # implicit layer similarity read over

        # read abstract-to-implicit api_name table
        print("### initializing maps")
        target_csv_path = p.join(self.database_path, "abstract", "abstract_api_name.csv")
        f = open(target_csv_path, "r")
        reader = csv.reader(f)
        row_list = []
        for row in reader:
            if len(row) > 2:
                row_list.append(row)
        head = row_list[0]
        row_list = row_list[1:]
        valid_locations = []
        for library in self.library_list:
            valid_locations.append(head.index(library))
        self.main_api_name_map = {}
        self.inverse_map = {}
        for library in self.library_list:
            self.inverse_map[library] = {}
        for row in row_list:
            api_id = row[0]
            abstract_api_name = row[1]
            implicit_library_api_name = []
            for location in valid_locations:
                implicit_library_api_name.append(row[location])
            dict_for_main_map = {"id": api_id}
            for i in range(len(self.library_list)):
                dict_for_main_map[self.library_dict[i]] = implicit_library_api_name[i]
            self.main_api_name_map[abstract_api_name] = dict_for_main_map
            for i in range(len(self.library_list)):
                self.inverse_map[self.library_list[i]][implicit_library_api_name[i]] = abstract_api_name
        # abstract-to-implicit api_name table read over

        self.threshold = 0.5

        # initializing candidate map
        print("### initializing candidate map")
        self.candidate_map = self.get_candidate_dict(self.threshold)
        # candidate map initialize over

        print("### Database OK")
        return

    def is_library_valid(self, library: str) -> bool:
        return library in self.library_list

    def is_abstract_api_name_valid(self, abstract_api_name: str) -> bool:
        return abstract_api_name in list(self.main_api_name_map.keys())

    def is_implicit_api_name_valid(self, library: str, implicit_api_name: str) -> bool:
        return self.is_library_valid(library) and \
            (implicit_api_name in implicit_api_name in list(self.inverse_map[library].keys()))

    def set_threshold(self, threshold: float) -> None:
        assert 0.01 <= threshold <= 0.99, "check threshold, between 0.01 and 0.99."
        self.threshold = threshold
        self.candidate_map = self.get_candidate_dict(self.threshold)
        print("Threshold changed to " + str(threshold) + ", map updated.")
        return

    def __get_all_api_list(self, library: str) -> list:
        assert self.is_library_valid(library), "check library name"
        return list(self.layer_info[library].keys())

    def __get_api_description(self, library: str, implicit_api_name: str) -> str:
        assert self.is_implicit_api_name_valid(library, implicit_api_name), "check input"
        flag = False
        for lib in self.library_list:
            if implicit_api_name.startswith(lib):
                pass
            else:
                flag = True
        if not flag:
            return self.layer_info[library][library + ".nn." + implicit_api_name]["descp"]
        else:
            return self.layer_info[library][implicit_api_name]["descp"]

    def get_api_similarity(self, library: str, implicit_api_name: str) -> dict:
        assert self.is_implicit_api_name_valid(library, implicit_api_name), "check input"
        return self.layer_similarity[library][implicit_api_name]

    def get_implicit_api_name(self, library: str, abstract_api_name: str) -> str:
        assert self.is_library_valid(library), "check input library"
        assert self.is_abstract_api_name_valid(abstract_api_name), "check input abstract_api_name"
        return self.main_api_name_map[abstract_api_name][library]

    def get_abstract_api_name(self, library: str, implicit_api_name: str) -> str:
        assert self.is_implicit_api_name_valid(library, implicit_api_name), "check input"
        return self.inverse_map[library][implicit_api_name]

    def get_candidate_mutate_list(self, abstract_api_name: str) -> list[str]:
        assert self.is_abstract_api_name_valid(abstract_api_name)
        return self.candidate_map[abstract_api_name]

    def __get_candidate_mutate_list(self, abstract_api_name: str, threshold: float) -> list[str]:
        assert abstract_api_name in list(self.main_api_name_map.keys()), "check input abstract_api_name"
        abstract_candidate_api_list = list(self.main_api_name_map.keys())
        for library in self.library_list:
            current_implicit_api_name = self.get_implicit_api_name(library, abstract_api_name)
            implicit_candidate_api_list = []
            current_abstract_candidate_api_list = []
            similarity_dict = self.get_api_similarity(library, current_implicit_api_name)
            for layer in similarity_dict.keys():
                if similarity_dict[layer] >= threshold and layer in self.inverse_map[library].keys():
                    implicit_candidate_api_list.append(layer)
            for layer in implicit_candidate_api_list:
                current_abstract_candidate_api_list.append(self.get_abstract_api_name(library, layer))
            abstract_candidate_api_list = merge(abstract_candidate_api_list, current_abstract_candidate_api_list)
        return abstract_candidate_api_list

    def __get_candidate_dict(self, threshold: float) -> (dict, int):
        zero_num = 0
        # abstract_api_num = len(self.main_api_name_map)
        result_dict = {}
        abstract_api_list = list(self.main_api_name_map.keys())
        for abstract_api_name in abstract_api_list:
            candidate_list = self.__get_candidate_mutate_list(abstract_api_name, threshold)
            if len(candidate_list) == 0:
                zero_num += 1
            result_dict[abstract_api_name] = candidate_list
        return result_dict, zero_num

    def get_candidate_dict(self, threshold: float) -> dict:
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

    def refresh(self) -> None:
        threshold = self.threshold
        self.__init__()
        self.set_threshold(threshold)
        return


d = Database()
# d.generate_candidate_report()
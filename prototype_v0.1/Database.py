import os
import os.path as p
import yaml


class Database:
    def __init__(self):
        self.database_path = p.join(".", "database")
        self.implicit_database_path = p.join(self.database_path, "implicit")

        print("### Database initializing...")

        # read config
        print("### initializing config")
        config_path = p.join(".", "config", "config.yaml")
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

        print("### Database OK")

    def get_all_api_list(self, library: str) -> list:
        assert library in self.library_list, "check library name"
        return list(self.layer_info[library].keys())

    def get_api_description(self, library: str, api_name: str) -> str:
        assert library in self.library_list and api_name in self.get_all_api_list(library), "check input"
        flag = False
        for lib in self.library_list:
            if api_name.startswith(lib):
                pass
            else:
                flag = True
        if not flag:
            return self.layer_info[library][library + ".nn." + api_name]["descp"]
        else:
            return self.layer_info[library][api_name]["descp"]

    def get_api_similarity(self, library: str, api_name: str) -> dict:
        assert library in self.library_list and api_name in self.get_all_api_list(library), "check input"
        return self.layer_similarity[library][api_name]


d = Database()

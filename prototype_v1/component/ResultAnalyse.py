import os.path as p
import yaml


def dict_to_string(dictionary):
    result = "\n".join([f"{key}: {value}" for key, value in dictionary.items()])
    return result


def generate_report(result_info: str, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result_info)
    return


class Analyser:
    def __init__(self):
        self.CONFIG_PATH = p.join("..", "config", "config.yaml")
        self.library_list = []
        self.refresh_config()
        return

    def refresh_config(self):
        f = open(self.CONFIG_PATH, "r", encoding="utf-8")
        config = yaml.full_load(f)
        f.close()
        self.library_list = list(config["LIBRARY_LIST"].values())

    def analyse_result(self, result_list: list[dict]) -> bool:
        report = ""
        for i in range(len(result_list)):
            library = self.library_list[i]
            report += library + ":\n\n"
            report += dict_to_string(result_list[i])
            report += "\n\n"
        report_path = result_list[0]["case path"].replace((self.library_list[0] + "_version.py"), "report.txt")
        generate_report(report, report_path)
        for result in result_list:
            if not result["run test"]:
                return False
            elif not result["train test"]:
                return False
            else:
                return True
        return True


analyser = Analyser()

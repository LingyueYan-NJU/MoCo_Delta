import os.path
import random


class Exec:
    def __init__(self):
        pass

    def translate_abs_case(self, case: dict) -> str:
        # return file name
        pass

    def test_run(self, model_file_name: str) -> bool:
        # run translated test case, and return if it ran successfully.
        pass

    def train(self, model_file_name: str) -> bool:
        pass

    def get_result(self, model, case_path) -> list[float]:
        # 基于numpy, list[运行是否通过(0/1), 训练是否通过(0/1), shape和, shape积, 元素和]
        pass


class Exec_pytorch(Exec):
    def translate_abs_case(self, case: dict) -> str:
        pass

    def test_run(self, model_file_name: str) -> bool:
        if random.randint(1, 20) < 2:
            return False
        else:
            return True

    def train(self, model_file_name: str) -> bool:
        if random.randint(1, 20) < 2:
            return False
        else:
            return True

    def get_result(self, model, case_path) -> list[float]:
        f = open(os.path.join(case_path, "pytorch_version.py"), "w", encoding="utf-8")
        f.write("# pytorch version!")
        f.close()
        result = [1.0 if self.test_run("test") else 0.0, 1.0 if self.train("test") else 0.0, 12.0, 12.0,
                  random.random()]
        return result


class Exec_paddle(Exec):
    def translate_abs_case(self, case: dict) -> str:
        pass

    def test_run(self, model_file_name: str) -> bool:
        if random.randint(1, 20) < 2:
            return False
        else:
            return True

    def train(self, model_file_name: str) -> bool:
        if random.randint(1, 20) < 2:
            return False
        else:
            return True

    def get_result(self, model, case_path) -> list[float]:
        f = open(os.path.join(case_path, "paddle_version.py"), "w", encoding="utf-8")
        f.write("# paddle version!")
        f.close()
        result = [1.0 if self.test_run("test") else 0.0, 1.0 if self.train("test") else 0.0, 12.0, 12.0,
                  random.random()]
        return result


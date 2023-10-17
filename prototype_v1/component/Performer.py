from _database import database
database = database


class Performer:
    def __init__(self):
        self.at = 1
        return

    def get_library_name(self) -> str:
        pass

    def translate(self, abstract_model: dict) -> str:
        # Given a dict of abstract model, translate it into a str which can be written into a .py file
        # And then this .py file will be a whole model.
        pass

    def get_model_from_file(self, file_path: str):
        # Given a file path of a .py file, turn the net in this file to a model in memory.
        # Return this model.
        pass

    def train(self, model) -> float:
        # Given a model, train it, and return time cost(if failed, return -1).
        pass

    def run(self, model) -> (float, float, list[int], str):
        # Given a model, give a tensor and run it, then
        # 1. return time cost(-1 if failed).
        # 2. return the calculate result(-1 if failed).
        # 3. return the shape([] if failed).
        # 4. return the error info("" if succeeded).
        pass

import random
from Performer import Performer, database


class TorchPerformer(Performer):
    # TODO
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "torch"

    def translate(self, abstract_model: dict) -> str:
        return "torch model code"

    def get_model_from_file(self, file_path: str):
        return "torch model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, float, list[int], str):
        return random.choice([1.0, 2.0, 3.0, -1]), 1.0, [1, 1, 1, 1], "error?"

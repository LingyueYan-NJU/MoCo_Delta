from Performer import Performer, database


class MindSporePerformer(Performer):
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "mindspore"

    def translate(self, abstract_model: dict) -> str:
        return "mindspore model code"

    def get_model_from_file(self, file_path: str):
        return "mindspore model"

    def train(self, model) -> float:
        return 342864.0

    def run(self, model) -> (float, float, list[int], str):
        return 342864.0, 342864.0, [1, 1, 1, 1], ""

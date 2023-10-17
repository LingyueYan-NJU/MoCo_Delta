from Performer import Performer, database


class TensorFlowPerformer(Performer):
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "tensorflow"

    def translate(self, abstract_model: dict) -> str:
        return "tensorflow model code"

    def get_model_from_file(self, file_path: str):
        return "tensorflow model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, float, list[int], str):
        return 1.0, 1.0, [1, 1, 1, 1], ""

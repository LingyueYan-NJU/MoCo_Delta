from Performer import Performer, database


class JittorPerformer(Performer):
    def __init__(self):
        super().__init__()
        return

    def get_library_name(self) -> str:
        return "jittor"

    def translate(self, abstract_model: dict) -> str:
        return "jittor model code"

    def get_model_from_file(self, file_path: str):
        return "jittor model"

    def train(self, model) -> float:
        return 1.0

    def run(self, model) -> (float, float, list[int], str):
        return 1.0, 1.0, [1, 1, 1, 1], ""

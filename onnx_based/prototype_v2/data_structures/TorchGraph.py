from TorchNode import TorchNode


class TorchGraph:
    def __init__(self):
        self.node_list: list[TorchNode] = []
        self.first_input_shape = []
        self.first_input_name = ""
        self.last_output_name = ""
        return

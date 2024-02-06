import torch
from Node import Node


class TorchNode(Node):
    def __init__(self):
        super().__init__()
        self.torch_layer = torch.nn.ReLU()
        return

    def to_define(self):
        define_sentence = f"{self.op_type}("
        for attr in self.attributes.values():
            define_sentence += f"{attr['name']} = {str(attr['value'])}, "
        if define_sentence.endswith(", "):
            define_sentence = define_sentence[:-2]
        define_sentence += ")"
        return f"self.{self.node_name.replace('/', '_')} = {define_sentence}"

    def to_forward(self):
        if isinstance(self.input_name, list):
            input_list = str(self.input_name)
            input_list = input_list[1:-1]
        else:
            input_list = self.input_name
        return f"{self.output_name.replace('/', '_')} = self.{self.node_name.replace('/', '_')}({input_list})"


if __name__ == "__main__":
    tn = TorchNode()
    tn.attributes = {'dilations': {'name': 'dilations', 'type': 'INTS', 'value': [1, 1]},
                     'groups': {'name': 'group', 'type': 'INT', 'value': 1},
                     'pads': {'name': 'pads', 'type': 'INTS', 'value': [0, 0]},
                     'strides': {'name': 'strides', 'type': 'INTS', 'value': [1, 1]},
                     'in_channels': {'name': 'in_channels', 'type': 'INT', 'value': 1},
                     'out_channels': {'name': 'out_channels', 'type': 'INT', 'value': 5},
                     'kernel_size': {'name': 'kernel_size', 'type': 'INTS', 'value': [3, 3]}}
    tn.torch_layer = torch.nn.Conv2d(1, 5, [3, 3], [1, 1], [0, 0], [1, 1], 1, False, "zeros", None, None)
    tn.input_name = "test_input"
    tn.node_name = "test_node_name"
    tn.output_name = "test_output"
    tn.op_type = "torch.nn.Conv2d"

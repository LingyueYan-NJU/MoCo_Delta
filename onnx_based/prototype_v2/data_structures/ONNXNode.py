from MoCo_Delta.onnx_based.prototype_v2.data_structures.Node import Node


class ONNXNode(Node):
    def __init__(self):
        super().__init__()
        self.initializers = {}
        return

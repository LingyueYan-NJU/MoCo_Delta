import copy

from ONNXNode import ONNXNode
import onnx


class ONNXGraph:
    def __init__(self):
        self.node_list: list[ONNXNode] = []
        self.first_input_name = ""
        self.last_output_name = ""
        return

    def read_file(self, file_path):
        self.node_list.clear()
        model = onnx.load(file_path)
        graph = model.graph
        nodes = []
        for node in graph.node:
            nodes.append(node)
        initializers = []
        for init in graph.initializer:
            initializers.append(init)
        for node in nodes:
            inputs = node.input
            current_node = ONNXNode()
            current_node.attributes.clear()
            current_node.initializers.clear()
            if len(inputs) == 0 or len(inputs) == 1:
                current_node.input_name = "" if len(inputs) == 0 else inputs[0]
            else:
                inputs_to_be_decided = copy.deepcopy(inputs)
                # current_node.input_name = inputs[0]
                for init in initializers:
                    if init.name in inputs:
                        current_node.initializers[init.name] = init
                        inputs_to_be_decided.remove(init.name)
                current_node.input_name = \
                    inputs_to_be_decided[0] if len(inputs_to_be_decided) == 1 else list(inputs_to_be_decided)
            current_node.output_name = node.output[0]
            current_node.node_name = node.name
            current_node.op_type = node.op_type
            for attr in node.attribute:
                new_attr = {}
                attr_name = attr.name
                new_attr["name"] = attr_name
                if attr.type == attr.INT:
                    new_attr["type"] = "INT"
                    new_attr["value"] = attr.i
                elif attr.type == attr.FLOAT:
                    new_attr["type"] = "FLOAT"
                    new_attr["value"] = attr.f
                elif attr.type == attr.INTS:
                    new_attr["type"] = "INTS"
                    new_attr["value"] = list(attr.ints)
                elif attr.type == attr.STRING:
                    new_attr["type"] = "STRING"
                    new_attr["value"] = str(attr.s)
                else:
                    print(f"{str(attr)}\nThis type of attribute has not been implemented.")
                    new_attr["type"] = "NOTIMPLEMENTED"
                    new_attr["value"] = "NONE"
                current_node.attributes[attr_name] = copy.deepcopy(new_attr)
            self.node_list.append(copy.deepcopy(current_node))
        self.first_input_name = self.node_list[0].input_name
        self.last_output_name = self.node_list[-1].output_name
        return

    def get_node(self, node_name):
        for node in self.node_list:
            if node.get_node_name() == node_name:
                return node
        print(f"Oh, no node named{node_name}")
        return None

    def translate(self, lib_name):
        return


if __name__ == "__main__":
    graph = ONNXGraph()
    graph.read_file("../../pre/onnx_test/testModel.onnx")

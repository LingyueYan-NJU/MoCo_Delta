import copy
from ONNXNode import ONNXNode
import onnx


class ONNXGraph:
    def __init__(self):
        self.node_list: list[ONNXNode] = []
        self.first_input_shape = []
        self.first_input_name = ""
        self.last_output_name = ""
        return

    def parse_onnx_model(self, onnx_model):
        # ONNXGraph.parse_onnx_model:
        # 输入一个onnx模型，该模型是通过onnx.load从文件中读取出来的。
        # 然后会分析模型，并以list[ONNXNode]的组织形式将该模型*存储到本ONNXGraph的壳中*。
        # 关于ONNXNode的细节，详见Node类与ONNXNode类。

        # 首先现在要读取新模型了，我们要清空本对象的node_list
        self.node_list.clear()

        # 提取输入形状，由于我们的全部种子模型一定只有一个四维输入，所以直接这样硬写入，不做判断。
        input_shape = []
        dim_datas = onnx_model.graph.input[0].type.tensor_type.shape.dim
        for dim_data in dim_datas:
            input_shape.append(dim_data.dim_value)
        self.first_input_shape = input_shape

        # 提取onnx_model.graph
        graph = onnx_model.graph

        # 从graph中提取结点和初始化器
        nodes = []
        for node in graph.node:
            nodes.append(node)
        initializers = []
        for init in graph.initializer:
            initializers.append(init)

        # 分析每一个结点，并将这个结点和它对应的初始化器，包装在ONNXNode中。
        for node in nodes:
            inputs = node.input

            # 新建并初始化一个ONNXNode
            current_node = ONNXNode()
            current_node.attributes.clear()
            current_node.initializers.clear()

            # 如果没有*input*，或只有一个*input*，那说明该结点没有初始化器，就直接存入input_name
            if len(inputs) == 0 or len(inputs) == 1:
                current_node.input_name = "" if len(inputs) == 0 else inputs[0]
            else:

                # 如果有多个 *input*，那么找出哪些对应的是初始化器，将对应的初始化器加入initializers，并在输入列表中删除该名称。
                inputs_to_be_decided = copy.deepcopy(inputs)

                for init in initializers:
                    if init.name in inputs:
                        current_node.initializers[init.name] = init
                        inputs_to_be_decided.remove(init.name)

                # 如果结点有多个 *input*，那么就存输入的名称的列表，否则存单个名称
                current_node.input_name = \
                    inputs_to_be_decided[0] if len(inputs_to_be_decided) == 1 else list(inputs_to_be_decided)

            # 存入 *output*
            current_node.output_name = node.output[0]

            # 存入 *name*
            current_node.node_name = node.name

            # 存入 *op_type*
            current_node.op_type = node.op_type

            # 根据属性的不同类别，存入 *attribute*
            # TODO: 类别现在不全，待补充。
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

            # 结点加入本对象node_list
            self.node_list.append(copy.deepcopy(current_node))

        # 设定结点最初的输入与最后的输出
        self.first_input_name = self.node_list[0].input_name
        self.last_output_name = self.node_list[-1].output_name
        return

    def read_file(self, file_path):
        # 输入文件路径，从文件中读取一个onnx模型，然后分析该模型，包装各结点为ONNXNode，并存入本对象的node_list。
        self.node_list.clear()
        model = onnx.load(file_path)
        self.parse_onnx_model(model)
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
    graph_ = ONNXGraph()
    graph_.read_file("../../pre/onnx_test/testModel.onnx")

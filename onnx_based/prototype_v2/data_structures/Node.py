class Node:
    def __init__(self):
        self.input_name = ""
        self.output_name = ""
        self.node_name = ""
        self.op_type = ""
        self.attributes = {}
        self.initializers = {}

    def get_input_name(self):
        return self.input_name

    def get_output_name(self):
        return self.output_name

    def get_node_name(self):
        return self.node_name

    def get_op_type(self):
        return self.op_type

    def get_attribute_value(self, attribute_name: str, default=None):
        if attribute_name in self.attributes.keys():
            return self.attributes[attribute_name]["value"]
        else:
            return default

    def get_attribute_type(self, attribute_name: str, default=None):
        if attribute_name in self.attributes.keys():
            return self.attributes[attribute_name]["type"]
        else:
            return default

    def translate(self, lib):
        return

import os.path
from html.parser import HTMLParser
from bs4 import BeautifulSoup
import requests


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, d):
        self.text.append(d)

    def get_data(self):
        return ''.join(self.text)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def extract_list_items(html):
    soup = BeautifulSoup(html, 'html.parser')
    list_items = soup.find_all('li')

    contents = [item.get_text() for item in list_items]
    return contents


def fetch_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 确保请求成功
        return response.text
    except requests.RequestException as e:
        return f"请求错误: {e}"


def get_url_torch(api_name):
    return f"https://pytorch.org/docs/stable/generated/{api_name}.html"


def get_url_onnx(api_name):
    return f"https://onnx.ai/onnx/operators/onnx__{api_name}.html"


# ==================
# FINAL METHOD TORCH
# ==================
def torch_handle(api_name):
    try:
        text = fetch_url_content(get_url_torch(api_name))
        para_text = text.split(
            '<dt class="field-odd">Parameters</dt>\n<dd class="field-odd"><ul class="simple">\n',
            1
        )[1].split(
            '</ul>\n</dd>\n</dl>\n',
            1
        )[0]
        para_text = strip_tags(para_text)
        para_texts = para_text.split("\n")

        res = []

        for line in para_texts:
            if "(" in line and ")" in line and " – " in line:
                para_name = line.split(" (", 1)[0]
                para_type = line.split("(", 1)[1].split(")", 1)[0].replace(", optional", "")
                para_des = line.split(" – ", 1)[1]
                res.append([para_name, para_type, para_des])
            else:
                if len(res) > 0:
                    res[-1][2] += (" " + line)
        return res
    except Exception:
        f_txt = open("./log.txt", "a", encoding="utf-8")
        f_txt.write(f"Please manually handle {api_name}.\n")
        f_txt.close()
        return []
# ==================
# FINAL METHOD TORCH
# ==================


def simpler(tmp_res):
    res = []
    for e in tmp_res:
        res.append([e[0], f"type: {e[1]}. {e[2]}"])
    return res


# onnx handle
def find_section_with_keyword(html, keyword):
    soup = BeautifulSoup(html, 'lxml')

    # 寻找所有的section标签
    sections = soup.find_all('section', id=lambda x: x and keyword in x)

    # 如果找到匹配的section标签，返回其内容
    if sections:
        return str(sections[0])
    else:
        return "NO"


# =================
# FINAL METHOD ONNX
# =================
def onnx_handle(api_name):

    res_attribute = []
    res_inputs = []

    text = fetch_url_content(get_url_onnx(api_name))
    text = find_section_with_keyword(text, api_name.lower())
    attributes = find_section_with_keyword(text, "attributes")
    inputs = find_section_with_keyword(text, "inputs")
    # type_constraints = find_section_with_keyword(text, "type-constraints")
    if attributes != "NO":
        attributes_list = extract_list_items(attributes)
        for line in attributes_list:
            if " - " not in line or ":" not in line:
                print(api_name + " . " + line + " has some problem")
                continue
            else:
                para_name_and_type, des = line.split(":", 1)[0], line.split(":", 1)[1][3:]
                para_name, para_type = para_name_and_type.split(" - ", 1)[0], para_name_and_type.split(
                    " - ", 1)[1].split("(", 1)[0][:-1]
                res_attribute.append([para_name, para_type, des])
    if inputs != "NO":
        inputs_list = extract_list_items(inputs)
        for line in inputs_list:
            if " - " not in line or ":" not in line:
                print(api_name + " . " + line + " has some problem")
                continue
            else:
                para_name_and_type, des = line.split(":", 1)[0], line.split(":", 1)[1].replace("\n", "").replace(
                    "\n", "")
                para_name, para_type = para_name_and_type.split(" - ", 1)[0], para_name_and_type.split(
                    " - ", 1)[1].split(
                    "(", 1)[0]
                res_inputs.append([para_name, para_type, des])
    return res_attribute, res_inputs
# =================
# FINAL METHOD ONNX
# =================


def tf_txt_analyse(api_name):

    res = []

    if not os.path.exists(f"./tensorflow/{api_name}.txt"):
        return []
    else:
        f = open(f"./tensorflow/{api_name}.txt", "r", encoding="utf-8")
        lines = f.readlines()
        f.close()
        for line in lines[1:]:
            para_name, para_info = line.split(":", 1)[0], line.split(":", 1)[1]
            if para_info.startswith(" "):
                para_info = para_info[1:]
            para_name = para_name.replace(" ", "_")
            res.append([para_name, para_info])

    return res


def generate():
    if os.path.exists("./para_info"):
        return
    os.makedirs("./para_info")
    import json
    f = open("./op_types_mapping_2.1.json", "r", encoding="utf-8")
    all_api_dict = json.load(f)
    f.close()
    for key in all_api_dict.keys():
        os.makedirs(f"./para_info/{str(key)}")
        os.makedirs(f"./para_info/{str(key)}/onnx")
        os.makedirs(f"./para_info/{str(key)}/onnx/attributes")
        os.makedirs(f"./para_info/{str(key)}/onnx/inputs")
        os.makedirs(f"./para_info/{str(key)}/pytorch")
        os.makedirs(f"./para_info/{str(key)}/tensorflow")
        current_onnx_api = str(key)
        attributes, inputs = onnx_handle(current_onnx_api)
        attributes, inputs = simpler(attributes), simpler(inputs)
        if len(attributes) > 0:
            for e in attributes:
                f = open(f"./para_info/{str(key)}/onnx/attributes/{e[0]}.txt", "w", encoding="utf-8")
                f.write(e[1])
                f.close()
                print(f"written ./para_info/{str(key)}/onnx/attributes/{e[0]}.txt")
        if len(inputs) > 0:
            for e in inputs:
                f = open(f"./para_info/{str(key)}/onnx/inputs/{e[0]}.txt", "w", encoding="utf-8")
                f.write(e[1])
                f.close()
                print(f"written ./para_info/{str(key)}/onnx/inputs/{e[0]}.txt")
        current_pytorch_api_list = all_api_dict[key]["pytorch"]["apis"]
        for current_pytorch_api in current_pytorch_api_list:
            os.makedirs(f"./para_info/{str(key)}/pytorch/{current_pytorch_api}")
            info = torch_handle(current_pytorch_api)
            info = simpler(info)
            if len(info) > 0:
                for e in info:
                    if "*" in e[0]:
                        continue
                    f = open(f"./para_info/{str(key)}/pytorch/{current_pytorch_api}/{e[0]}.txt", "w", encoding="utf-8")
                    f.write(e[1])
                    f.close()
                    print(f"written ./para_info/{str(key)}/pytorch/{current_pytorch_api}/{e[0]}.txt")
        current_tensorflow_api_list = all_api_dict[key]["tensorflow"]["apis"]
        for current_tensorflow_api in current_tensorflow_api_list:
            os.makedirs(f"./para_info/{str(key)}/tensorflow/{current_tensorflow_api}")
            info = tf_txt_analyse(current_tensorflow_api)
            if len(info) > 0:
                for e in info:
                    if "*" in e[0]:
                        continue
                    f = open(f"./para_info/{str(key)}/tensorflow/{current_tensorflow_api}/{e[0]}.txt", "w",
                             encoding="utf-8")
                    f.write(e[1])
                    f.close()
                    print(f"written ./para_info/{str(key)}/tensorflow/{current_tensorflow_api}/{e[0]}.txt")


generate()

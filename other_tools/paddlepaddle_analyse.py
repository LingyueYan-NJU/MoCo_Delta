# install requests, beautifulsoup4
import requests
from bs4 import BeautifulSoup
from handle_page import handle_page
import yaml
import os


def get_api_name(info: str):
    return "paddle.nn." + info.split("_en.html")[0].split("paddle/nn/")[1]


file_path = "./../paddle_layer_info"
if not os.path.exists(file_path):
    os.makedirs(file_path)
main_url = "https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html"
original_url = "https://www.paddlepaddle.org.cn"
# response = requests.get(main_url)
# content = ""
# if response.status_code == 200:
#     content = str(response.content)
# else:
#     pass
# soup = BeautifulSoup(response.text, "html.parser")
# lists = soup.find_all("li")
# nn_name_li = []
# for ll in lists:
#     s = str(ll)
#     if ("paddle/nn" not in s) or ("toctree-l3" not in s):
#         continue
#     elif len(s) >= 300:
#         continue
#     else:
#         nn_name_li.append(get_api_name(s))


# get api list
zh_page = "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Overview_cn.html"
template_page = "https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/TEMPLATE_en.html"
response = requests.get(zh_page)
soup = BeautifulSoup(response.text, "html.parser")
tables = soup.find_all(class_="section")
tables = tables[1:]
accept_list = ["卷积层", "pooling 层", "Padding 层", "激活层", "Normalization 层", "循环神经网络层",
               "Transformer 相关", "线性层", "Dropout 层", "Embedding 层", "Vision 层",
               "Clip 相关", "公共层"]
layer_name_list = []
# check h2
for table in tables:
    h2 = str(table).split("<h2>")[1].split("<")[0]
    print(h2)
    if h2 not in accept_list:
        continue
    tbody = table.find_all("tbody")[0]
    trs = tbody.find_all("tr")
    for tr in trs:
        tr_str = str(tr)
        name = tr_str.split("paddle.nn.", 1)[1].split("<")[0]
        layer_name_list.append("paddle.nn." + name)
for name in layer_name_list:
    page = template_page.replace("TEMPLATE", name.replace('paddle.nn.', ''))
    try:
        d = handle_page(page)
        f = open(os.path.join(file_path, name + ".yaml"), "w", encoding="utf-8")
        yaml.dump(d, f)
        f.close()
        print(name + " OK")
    except Exception:
        print(name + " not OK")

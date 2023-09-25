import requests
from bs4 import BeautifulSoup

test_page = 'https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/Conv1D_en.html'


def handle_page(page_url: str):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, "html.parser")
    # define
    a = soup.find_all(class_="sig sig-object py")
    api_define_sentence = a[0].text.replace("\n", "")
    a = soup.find_all(class_="field-list simple")
    para_description_sentence = a[0].text.replace("\n", "")
    p2 = para_description_sentence.replace(". ", "|||||")
    plist = p2.split(".")
    for i in range(len(plist)):
        plist[i] = plist[i].replace("|||||", ". ")
    plist[0] = plist[0][31:]
    a = soup.find_all("dd")[0]
    aa = a.find_all("p")[0]
    description_sentence = aa.text
    api_define_sentence = api_define_sentence.replace(" ", "").replace("[source]", "")[5:]
    result_dict = create_empty_full_dict()
    result_dict["api"] = "paddle.nn." + page_url.split("paddle/nn/", 1)[1].split("_en", 1)[0]
    result_dict["descp"] = description_sentence
    para_sentence = api_define_sentence.replace("paddle.nn.", "")[len(result_dict["api"])-10:]
    para_dict = get_params_simple(para_sentence)
    para_name_list = list(para_dict.keys())
    for para_name in para_name_list:
        if para_dict[para_name] == "Required":
            result_dict["inputs"]["required"].append(para_name)
        else:
            result_dict["inputs"]["optional"].append(para_name)
    for para_name in para_name_list:
        for sentence in plist:
            if sentence.startswith(para_name):
                result_dict["constraints"][para_name] = get_para_info(para_name, para_dict[para_name], sentence)
                # test
                # print("get_para_info(" + para_name + ", " + para_dict[para_name] + ", " + sentence + ")")
                break
    return result_dict


def create_empty_full_dict() -> dict:
    result = {"api": None, "constraints": {}, "descp": None, "inputs": {"optional": [], "required": []}}
    return result


def create_empty_para_dict() -> dict:
    result = {"descp": None, "default": None, "dtype": None, "structure": None, "shape": None, "range": None,
              "enum": None}
    return result


def get_params_simple(line: str) -> dict:
    result_dict = {}
    ana = line.split('(', 1)[1][:-1]
    ana_list = ana.split(",")
    flag = False
    for param in ana_list:
        if "=" not in param and not flag:
            result_dict[param] = "Required"
        else:
            flag = True
            result_dict[param.split("=", 1)[0]] = param.split("=", 1)[1]
    return result_dict


def get_para_info(para_name: str, default, para_sentence: str) -> dict:
    result_dict = create_empty_para_dict()
    result_dict["default"] = default if default != "Required" else None
    define_part, description_part = para_sentence.split(" – ", 1)[0], para_sentence.split(" – ", 1)[1]
    define_part = define_part.replace(" ", "").\
        replace(para_name, "").replace(",optional", "").replace("(", "").replace(")", "")
    result_dict["descp"] = description_part
    result_dict["structure"] = define_part.split("|")
    result_dict["dtype"] = []
    result_dict["enum"] = ["need", "need", "need", "need"]
    if "str" in result_dict["structure"] and len(result_dict["structure"]) == 1:
        result_dict["dtype"].append("str")
    elif "str" in result_dict["structure"] and len(result_dict["structure"]) != 1:
        result_dict["dtype"].append("str")
        result_dict["dtype"].append("int")
    else:
        result_dict["dtype"].append("int")
        result_dict["enum"] = None
    if "tuple" in result_dict["structure"] or "list" in result_dict["structure"]:
        result_dict["shape"] = 2
    return result_dict


if __name__ == "__main__":
    d = handle_page(test_page)
    print(d)

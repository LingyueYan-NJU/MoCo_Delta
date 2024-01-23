import os
import pandas as pd


base_path = "./para_info"
total_para_name_to_api_name_and_para_des_dict = {}


def write_dict_to_excel(data, file_path):
    # 准备数据
    prepared_data = []
    for key, value_list in data.items():
        for inner_list in value_list:
            prepared_data.append([key, inner_list[0], inner_list[1]])

    # 转换为DataFrame
    df = pd.DataFrame(prepared_data, columns=["Key", "Value1", "Value2"])

    # 写入Excel
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

        # 调整Excel格式
        # workbook = writer.book
        # worksheet = writer.sheets['Sheet1']
        # current_row = 1
        # for key, values in data.items():
        #     next_row = current_row + len(values)
        #     if len(values) > 1:
        #         worksheet.merge_cells(start_row=current_row, start_column=1, end_row=next_row - 1, end_column=1)
        #     current_row = next_row


def get_all_para_name_and_api_name_and_para_des_from_a_folder(folder_path, lib):
    para_name_and_api_name_and_para_des = []
    if lib == "onnx":
        resolve_path = f"{folder_path}/onnx/attributes"
        file_list = os.listdir(resolve_path)
        api_name = folder_path.split("/")[-1]
        for file in file_list:
            para_name = file.replace(".txt", "")
            f = open(f"{resolve_path}/{file}", "r", encoding="utf-8")
            para_des = f.read()
            f.close()
            para_name_and_api_name_and_para_des.append([para_name, api_name, para_des])
        return para_name_and_api_name_and_para_des
    else:
        resolve_path = f"{folder_path}/{lib}"
        api_name_list = os.listdir(resolve_path)
        for api_name in api_name_list:
            file_list = os.listdir(f"{resolve_path}/{api_name}")
            for file in file_list:
                para_name = file.replace(".txt", "")
                f = open(f"{resolve_path}/{api_name}/{file}", "r", encoding="utf-8")
                para_des = f.read()
                f.close()
                para_name_and_api_name_and_para_des.append([para_name, api_name, para_des])
        return para_name_and_api_name_and_para_des


def expend_the_dict_for_a_triple_info_list(para_name_and_api_name_and_para_des_list):
    for para_name_and_api_name_and_para_des in para_name_and_api_name_and_para_des_list:
        para_name, api_name, para_des = para_name_and_api_name_and_para_des
        if para_name not in total_para_name_to_api_name_and_para_des_dict.keys():
            total_para_name_to_api_name_and_para_des_dict[para_name] = []
        total_para_name_to_api_name_and_para_des_dict[para_name].append([api_name, para_des])


def handle(lib):
    total_para_name_to_api_name_and_para_des_dict.clear()
    path_list = os.listdir(base_path)
    for path in path_list:
        target_path = f"{base_path}/{path}"
        expend_the_dict_for_a_triple_info_list(get_all_para_name_and_api_name_and_para_des_from_a_folder(
            target_path, lib))
    write_dict_to_excel(total_para_name_to_api_name_and_para_des_dict, f"./{lib}.xlsx")

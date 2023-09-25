from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import os
import numpy

ABS_LAYER_INFO_PATH = "./Abs_Layer_Info"
ABS_LAYER_SIMILARITY_PATH = "./Abs_Layer_Similarity"


def calculate_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    documents = [text1, text2]
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    similarity_score = similarity_matrix[0][0]
    return similarity_score


name_to_descp_dict = {}
abs_layer_list = []
temp = os.listdir(ABS_LAYER_INFO_PATH)
for abs_file_name in temp:
    abs_layer_list.append(abs_file_name[:-5])
for abs_layer_name in abs_layer_list:
    f = open(os.path.join(ABS_LAYER_INFO_PATH, abs_layer_name + ".yaml"), "r", encoding="utf-8")
    d = yaml.full_load(f)
    f.close()
    if "descp" not in d.keys() or d["descp"] is None:
        name_to_descp_dict[abs_layer_name] = abs_layer_name + " No Description No Description"
    else:
        name_to_descp_dict[abs_layer_name] = abs_layer_name + " " + d["descp"]

for abs_layer_name in abs_layer_list:
    similarity_dict = {}
    for abs_layer_name_2 in abs_layer_list:
        if abs_layer_name == abs_layer_name_2:
            continue
        similarity_value = calculate_similarity(name_to_descp_dict[abs_layer_name],
                                                name_to_descp_dict[abs_layer_name_2])
        similarity_value = numpy.float(similarity_value)
        similarity_value = round(similarity_value, 4)
        if similarity_value < 0.01:
            similarity_value = 0.01
        if similarity_value > 0.95:
            similarity_value = 0.95
        similarity_dict[abs_layer_name_2] = similarity_value
    f = open(os.path.join(ABS_LAYER_SIMILARITY_PATH, abs_layer_name + ".yaml"), "w", encoding="utf-8")
    yaml.dump(similarity_dict, f)
    f.close()
    print(abs_layer_name + " similarity written OK.")

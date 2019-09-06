# -*- coding: utf-8 -*-
# @CreatTime    : 2019/8/23 20:31
# @Author  : JasonLiu
# @FileName: data_analysis_process.py
"""
(1)对原始数据进行分析。文本的长度分别情况，每个类别的数据情况。
(2)对原始数据进行预处理，去掉停用词
"""
import json
import pdb
import pandas as pd
import numpy as np
import os
import jieba
import collections
import data_process_config as cfig
from sklearn.model_selection import train_test_split

labor_path = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/data/CAIL2019-FE-big/labor/train_selected_shoudong_modify.json"
loan_path = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/data/CAIL2019-FE-big/loan/train_selected.json"
divorce_path = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/data/CAIL2019-FE-big/divorce/train_selected.json"

stop_word_path = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/data/"
stop_word_filename = "stopwords_modify.txt"
stopwords_modify_path = os.path.join(stop_word_path, stop_word_filename)
stopwords_modify = open(stopwords_modify_path, mode='r').readlines()
stopwords_modify = [word.strip() for word in stopwords_modify]


def clean_data(name):
    """
    不该把所有标点符号去掉？
    :param name:
    :return:
    """
    setlast = jieba.lcut(name, cut_all=False)
    # stopwords_modify 或者 stopwords
    seg_list = [i.lower() for i in setlast if i not in stopwords_modify]  # 这里的停用词是不一样的，注意
    return "".join(seg_list)


def line_process(linedata, max_len, is_use_stop_words=True):
    # 是否去掉首句，是否使用停用词表？？
    data_str = linedata.replace("\n", "")
    if is_use_stop_words:
        data_str = clean_data(data_str)
    if len(data_str) > max_len:
        max_len = len(data_str)
    return data_str, max_len


def get_max_len(file_path):
    fin = open(file_path, 'r', encoding='utf8')
    line = fin.readline()
    raw_max_len = 0
    count = 0
    while line:
        d = json.loads(line)
        for sent in d:
            sentence = sent['sentence']
            labels = sent['labels']
            # if len(sentence) > 512:
            #     continue
            if len(sentence) > raw_max_len:
                raw_max_len = len(sentence)
            if len(sentence) > 512:
                count = count + 1
                print("sentence=", sentence)
                print("label=", labels)
        line = fin.readline()
    fin.close()
    return raw_max_len, count


def data_length_analysis():
    #
    labor_max_len, labor_num, = get_max_len(labor_path)
    loan_max_len, loan_num = get_max_len(loan_path)
    divorce_max_len, divorce_num = get_max_len(divorce_path)
    print("max len of labor={},more 512 num={}".format(labor_max_len, labor_num))
    print("max len of loan={},more 512 num={}".format(loan_max_len, loan_num))
    print("max len of divorce={},more 512 num={}".format(divorce_max_len, divorce_num))
    """
    max len of labor= 1104
    max len of loan= 775
    max len of divorce= 488
    """


def convert_list_csv(text_list, label_list, titles, sentence_label_dict, prefix=""):
    """
    将list形式的数据转为csv格式的数据。此外还生成一份json格式的test data
    :param text_list:
    :param label_list:
    :param titles:
    :param sentence_label_dict:
    :return:
    """

    # 将上述的句子和labels进行拼接
    label_array = np.array(label_list)
    orign_text_array = np.array(text_list)  # 不带分词
    orign_text_array = orign_text_array.reshape(orign_text_array.shape[0], -1)
    print(orign_text_array.shape)
    print(label_array.shape)
    orign_whole_data_array = np.concatenate([orign_text_array, label_array], axis=1)
    orign_whole_data = pd.DataFrame(orign_whole_data_array, columns=titles)

    orign_labor_csv_filename = os.path.join(cfig.data_base_dir, "{}train_test_data.csv".format(prefix))# 没有test data,用于最后的训练

    orign_whole_data.to_csv(orign_labor_csv_filename, sep="\t", encoding="utf-8",
                            index_label="id")  # index=0,则不保留行索引,header=0则不保存列名

    train_data, test_data = train_test_split(orign_whole_data, random_state=42, train_size=cfig.train_val_ratio,
                                             shuffle=True)

    train_filename = "{}train.csv".format(prefix)
    test_filename = "{}test.csv".format(prefix)
    test_json_file = "{}test.json".format(prefix)

    train_path = os.path.join(cfig.data_base_dir, train_filename)  # dev set是从train中再划分出0.1的数据作为dev set
    test_path = os.path.join(cfig.data_base_dir, test_filename)
    train_data.to_csv(train_path, sep="\t", encoding="utf-8", index_label="id")
    test_data.to_csv(test_path, sep="\t", encoding="utf-8", index_label="id")

    # 还需要生成test.json格式，方便与PyTorch版本对比，也方便采用官方的judger.py进行评估
    test_json_file = os.path.join(cfig.data_base_dir, test_json_file)
    test_list = []
    for (i, row) in enumerate(test_data.values):
        sentence = row[0]  # 注意，这里不是从文件中读取csv,所以没有index这一列
        # pdb.set_trace()
        labels = sentence_label_dict[sentence]
        # 查找对应的label names
        test_dict = {}
        test_dict["sentence"] = sentence
        test_dict["labels"] = labels
        test_list.append(test_dict)
    with open(test_json_file, 'w', encoding="utf-8") as out_test_json_file:
        json.dump(test_list, out_test_json_file, ensure_ascii=False)


def process_data():
    """
    对原始数据进行预处理，并划分出train data 和 test data
    :return:
    """
    raw_file_path = ""
    tag_file = ""

    tag_dic, tagname_dic = get_tag_dict(tag_file)
    LABEL_COLUMNS = []
    for t in tagname_dic:
        LABEL_COLUMNS.append(tagname_dic[t])
    titles = ["sentence"]
    titles.extend(LABEL_COLUMNS)

    fin = open(raw_file_path, 'r', encoding='utf8')
    line = fin.readline()
    sentence_tag_dict = collections.OrderedDict()
    while line:
        d = json.loads(line)
        for sent in d:
            sentence = sent['sentence'].strip()
            labels = sent['labels']  # LB1 - LB20
            if sentence in sentence_tag_dict:
                if sentence_tag_dict[sentence] != labels:
                    #                 print("same sentence={},label_pre={},label_now={}".format(sentence, sentence_tag_dict[sentence], labels))
                    # 对于不一致的情况，选择第一个labels有标注结果的即可。已有的是空，而新的非空，则覆盖。已有的非空，则不替换。
                    if not sentence_tag_dict[sentence]:
                        sentence_tag_dict[sentence] = labels
            else:
                sentence_tag_dict[sent['sentence']] = labels
        line = fin.readline()
    fin.close()

    # 遍历sentence_tag_dict
    alltext = []
    tag_label = []
    processed_max_len = 0
    prefix_name = ""
    for sentence in sentence_tag_dict:
        # 在此可以加一个文本预处理操作，比如去掉停用词
        if cfig.do_raw_text_process:
            # 进行文本的预处理
            prefix_name = "processed_"
            sentence, processed_max_len = line_process(sentence, processed_max_len, is_use_stop_words=False)
        alltext.append(sentence)
        labels = sentence_tag_dict[sentence]
        taglist = [0] * 20
        if labels:
            for i in labels:
                temp = tag_dic[i]
                taglist[temp] = 1
        tag_label.append(taglist)

    print("len alltext=", len(alltext))  # 未去重前， 5682条
    print("len tag_label=", len(tag_label))
    print("processed_max_len=", processed_max_len)  # 预处理后的文本最大长度
    alltext_set = set(alltext)
    print("去重后的样本数=", len(alltext_set))  # 还是否有重复的呢？
    # text_list, label_list, titles, sentence_label_dict

    convert_list_csv(alltext, tag_label, titles, sentence_tag_dict, prefix_name)


def get_tag_dict(tags_path):
    f = open(tags_path, 'r', encoding='utf8')
    tag_dic = {}
    tagname_dic = {}
    line = f.readline()
    while line:
        tagname_dic[len(tag_dic)] = line.strip()
        tag_dic[line.strip()] = len(tag_dic)
        line = f.readline()
    f.close()
    return tag_dic, tagname_dic


def data_csv_to_json(test_data_path, tag_file, out_file):
    """
    将分割出来的test.csv转为json格式，方便PyTorch版本的处理
    :return:
    """
    test_df = pd.read_csv(test_data_path, delimiter="\t")
    test_list = []
    tag_dic, tagname_dic = get_tag_dict(tag_file)
    for (i, row) in enumerate(test_df.values):
        guid = row[0]
        sentence = row[1]
        labels = row[2:]  # 此时labels类型是array
        # 查找对应的label names
        indexs = np.where(labels == 1)
        temp_labels = []
        if len(indexs[0]) > 0:
            for i in indexs[0]:
                temp_labels.append(tagname_dic[i])  # 查找到对应的label name
        test_dict = {}
        test_dict["sentence"] = sentence
        test_dict["labels"] = temp_labels
        test_list.append(test_dict)
    # 写到磁盘
    json.dump(test_list, out_file, ensure_ascii=False)


def process_csv_to_json(task_name):
    data_dir = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/bert_tensorflow_multi_label/data"
    data_path = os.path.join(data_dir, task_name)
    data_path = os.path.join(data_path, "test.csv")
    tag_file = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/data/CAIL2019-FE-big/{}/tags.txt".format(task_name)
    out_file = "./data/{}/test.json".format(task_name)
    out_f = open(out_file, 'w', encoding="utf-8")
    data_csv_to_json(data_path, tag_file, out_f)


def get_law_data_text_label_dict(raw_file_path, is_text_process=False):
    fin = open(raw_file_path, 'r', encoding='utf8')
    line = fin.readline()
    sentence_tag_dict = collections.OrderedDict()
    too_short_num = 0
    processed_max_len = 0
    raw_max_len = 0
    while line:
        d = json.loads(line)
        for sent in d:
            sentence = sent['sentence'].strip()
            if len(sentence) > raw_max_len:
                raw_max_len = len(sentence)
            if is_text_process:
                sentence, processed_max_len = line_process(sentence, processed_max_len, is_text_process)
            if len(sentence) < 3:
                too_short_num = too_short_num + 1
                continue
            labels = sent['labels']  # LB1 - LB20
            if sentence in sentence_tag_dict:
                if sentence_tag_dict[sentence] != labels:
                    if not sentence_tag_dict[sentence]:
                        sentence_tag_dict[sentence] = labels
            else:
                sentence_tag_dict[sent['sentence']] = labels
        line = fin.readline()
    fin.close()
    print("too_short_num=", too_short_num)
    print("raw_max_len=", raw_max_len)
    print("processed_max_len=", processed_max_len)
    return sentence_tag_dict


def merge_stages_data():
    """
    融合阶段1和阶段2的数据集。是否有新的数据样本在其中。
    :return:
    """
    stage_1_path = cfig.stage_1_path
    stage_2_path = cfig.stage_2_path
    # 分析阶段1的数据
    data_name = cfig.task_type_name  # 需要对3种数据进行训练： loan ， divorce ， labor
    num_labels = 20
    print("Start process {}".format(data_name))
    # 原始数据
    raw_data_stage1_dir = os.path.join(stage_1_path, data_name)
    raw_data_stage2_dir = os.path.join(stage_2_path, data_name)
    raw_data_stage1_filename = "data_small_selected.json"
    if data_name == "labor":
        # raw_data_stage2_filename = "train_selected.json"  # 对于labor数据集建议采用修正过的数据，因为原始的数据有些垃圾信息。
        raw_data_stage2_filename = "train_selected_shoudong_modify.json"  #
        print("raw_data_stage2_filename=", raw_data_stage2_filename)
    else:
        raw_data_stage2_filename = "train_selected.json"
    tag_file = os.path.join(raw_data_stage2_dir, "tags.txt")  # 在数据融合，生成csv阶段用到
    tag_dic, tagname_dic = get_tag_dict(tag_file)
    label_columns = []
    for t in tagname_dic:
        label_columns.append(tagname_dic[t])
    titles = ["sentence"]
    titles.extend(label_columns)

    # 是否对文本进行预处理
    is_text_process = True
    raw_stage1_file = os.path.join(raw_data_stage1_dir, raw_data_stage1_filename)
    raw_stage2_file = os.path.join(raw_data_stage2_dir, raw_data_stage2_filename)
    stage1_text_labels_dict = get_law_data_text_label_dict(raw_stage1_file, is_text_process)
    stage2_text_labels_dict = get_law_data_text_label_dict(raw_stage2_file, is_text_process)
    print("stage1_text_labels_dict len=", len(stage1_text_labels_dict))
    print("stage2_text_labels_dict len=", len(stage2_text_labels_dict))
    repeat_count = 0
    new_add_count = 0
    modify_count = 0
    labels_conflict = 0
    confuse_count = 0
    for text in stage1_text_labels_dict:
        # 是否在stage1中已经存在
        # pdb.set_trace()
        if text in stage2_text_labels_dict:
            repeat_count = repeat_count + 1
            # 如果出现重复，检测2个阶段的标签是否相同
            if set(stage1_text_labels_dict[text]) != set(stage2_text_labels_dict[text]):
                labels_conflict = labels_conflict + 1
                # 当出现冲突时
                if not stage2_text_labels_dict[text] and stage1_text_labels_dict[text]:
                    #当stage2为空，stage1非空，则用stage1
                    stage2_text_labels_dict[text] = stage1_text_labels_dict[text]
                    modify_count = modify_count + 1
                elif stage2_text_labels_dict[text] and not stage1_text_labels_dict[text]:
                    # stage2有，stage1为空
                    pass
                else:
                    # 都非空，不等,以stage2为准
                    # pdb.set_trace()
                    confuse_count = confuse_count + 1
                    # print("repeat sentence={},label_1={},label_2={}".format(text,
                    #                                                         stage1_text_labels_dict[text],
                    #                                                         stage2_text_labels_dict[text]))

        else:
            stage2_text_labels_dict[text] = stage1_text_labels_dict[text]
            new_add_count = new_add_count + 1
    print("repeat_count=", repeat_count)
    print("new_add_count=", new_add_count)
    print("labels_conflict=", labels_conflict)
    print("modify_count=", modify_count)
    print("confuse_count=", confuse_count)
    print("new stage2_text_labels_dict len=", len(stage2_text_labels_dict))
    # 将stage2_text_labels_dict分割为train data 和 test data

    # 转为text_list 和labels_list
    text_list = []
    labels_list = []
    prefix_name = "merge_processed_"  # 表示数据融合，且预处理
    for sentence in stage2_text_labels_dict:
        # 在此可以加一个文本预处理操作，比如去掉停用词
        text_list.append(sentence)
        labels = stage2_text_labels_dict[sentence]
        taglist = [0] * 20
        if labels:
            for i in labels:
                temp = tag_dic[i]
                taglist[temp] = 1
        labels_list.append(taglist)

    print("len alltext=", len(text_list))  # 未去重前， 5682条
    print("len tag_label=", len(labels_list))
    alltext_set = set(text_list)
    print("去重后的样本数=", len(alltext_set))  # 还是否有重复的呢？
    convert_list_csv(text_list, labels_list, titles, stage2_text_labels_dict, prefix_name)


def analysis_merge_data():
    data_name = cfig.task_type_name
    raw_data_stage2_dir = os.path.join(cfig.stage_2_path, data_name)
    #
    prefix_name = "merge_processed_"

    merge_train_file = os.path.join(raw_data_stage2_dir, prefix_name + "train_test_data.csv")
    train = pd.read_csv(merge_train_file, delimiter="\t")
    # 统计各个label类别的样本数
    df_labels = train.drop(['id', 'sentence'], axis=1)
    counts = []
    categories = list(df_labels.columns.values)
    label_count_dict = {}
    for i in categories:
        counts.append((i, df_labels[i].sum()))
        label_count_dict[i] = df_labels[i].sum()
    df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
    print(df_stats)  # 各个label的样本个数
    print("各个标签的样本数：")
    sort_label_count_dict = sorted(label_count_dict.items(), key=lambda item: item[1], reverse=True)
    print(sort_label_count_dict)
    """
    以下是train data的样本情况
    [('LB1', 1915), ('LB2', 1360), ('LB3', 1336), ('LB4', 944), ('LB5', 879), ('LB6', 860), ('LB8', 764), ('LB7', 756), 
    ('LB9', 536), ('LB10', 432), ('LB12', 56), ('LB18', 35), ('LB16', 34), ('LB11', 27), ('LB13', 27), ('LB20', 21), 
    ('LB14', 20), ('LB19', 12), ('LB15', 11), ('LB17', 2)]
    
    [('LN1', 1954), ('LN3', 1452), ('LN5', 1446), ('LN2', 1397), ('LN4', 1367), ('LN6', 1223), ('LN9', 1202), 
    ('LN7', 1033), ('LN8', 1027), ('LN10', 935), ('LN11', 87), ('LN15', 44), ('LN16', 30), ('LN13', 28), ('LN18', 26), 
    ('LN14', 24), ('LN12', 24), ('LN17', 14), ('LN19', 13), ('LN20', 9)]
    
    [('DV1', 6799), ('DV2', 4363), ('DV3', 3335), ('DV4', 1696), ('DV5', 1406), ('DV6', 1124), ('DV7', 1046),
     ('DV8', 1013), ('DV9', 996), ('DV10', 817), ('DV12', 496), ('DV13', 428), ('DV11', 340), ('DV14', 286), 
     ('DV19', 202), ('DV16', 191), ('DV20', 189), ('DV15', 175), ('DV18', 164), ('DV17', 122)]
    """


if __name__ == '__main__':
    # data_length_analysis()  # 分析原始数据的长度
    # process_data()  # 对原始数据进行预处理
    # merge_stages_data()  # 将第一阶段和第2阶段的数据集进行融合
    analysis_merge_data()  # 分析融合后数据基本信息，各个类别的样本数量

    # process_csv_to_json("labor")
    # process_csv_to_json("loan")
    # process_csv_to_json("divorce")

    # 在此基础上进行模型的评估
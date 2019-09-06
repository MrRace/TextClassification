# -*- coding: utf-8 -*-
# @CreatTime    : 2019/7/28 17:11
# @Author  : JasonLiu
# @FileName: data_process.py

import json
import os
import numpy as np
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split


PhaseNum = 2# 第1阶段，数据量较少
law_data_name = "loan"  # 需要对3种数据都进行增强。loan ， divorce ， labor
if PhaseNum == 1:
    data_dir = "../data/" + law_data_name
    raw_data_filename = "data_small_selected.json"
else:
    data_dir = "../data/CAIL2019-FE-big/" + law_data_name
    raw_data_filename = "train_selected.json"

filename = os.path.join(data_dir, raw_data_filename)
tag_file = os.path.join(data_dir, "tags.txt")
print("filename=", filename)
print("tag_file=", tag_file)
fin = open(filename, 'r', encoding='utf8')


def get_label_id(my_dic, q):
    return my_dic[q]


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


tag_dic, tagname_dic = get_tag_dict(tag_file)#构建映射词典，包括labe-id，id-label。这里id从0开始。而Label从LB1开始
task_cnt = 20

line = fin.readline()
import collections
sentence_tag_dict = collections.OrderedDict()
while line:
    d = json.loads(line)
    for sent in d:
        sentence = sent['sentence']
        labels = sent['labels']#LB1 - LB20

        if sentence in sentence_tag_dict:
            if sentence_tag_dict[sentence] != labels:
#                 print("same sentence={},label_pre={},label_now={}".format(sentence, sentence_tag_dict[sentence], labels))
                # 对于不一致的情况，选择第一个labels有标注结果的即可。已有的是空，而新的非空，则覆盖。已有的非空，则不替换。
                if not sentence_tag_dict[sentence]:
                    sentence_tag_dict[sentence] = labels
#                     print("wap,sent=", sentence)
        else:
            sentence_tag_dict[sent['sentence']] = labels
    line = fin.readline()
fin.close()
print("before augmentation,sentence_tag_dict len=", len(sentence_tag_dict))
# 从已经增强的数据中读取，对sentence_tag_dict进行扩充
augmentation_file_path = os.path.join(data_dir, "augment_text_labels_m1.txt")
repeat_count = 0
with open(augmentation_file_path, 'r', encoding="utf-8") as aug_fin:
    for aug_line in aug_fin.readlines():
        line_data = aug_line.strip()
        line_data = line_data.split("\t")
        s = line_data[0]
        s = s.strip()
        label = line_data[1]
        label = label.strip()
        # if not label in tag_dic:
        #     # print("new label,label=", label)

        if s in sentence_tag_dict:
            repeat_count = repeat_count + 1
            # 因为生产增强数据的时候也会将原来的数据添加进入
        else:
            # 需要将labels转为["LB1","LB2"]这种list
            lable_list = label.split(",")
            sentence_tag_dict[s] = lable_list
            # pdb.set_trace()
            # print("label=", label)
print("repeat_count=", repeat_count)
print("after augmentation,sentence_tag_dict len=", len(sentence_tag_dict))
# 所以按照道理，应该不会出现样本重复的情况。

# 遍历sentence_tag_dict
alltext = []
tag_label = []
for sentence in sentence_tag_dict:
    alltext.append(sentence.strip())
    labels = sentence_tag_dict[sentence]
    taglist = [0] * 20
    if labels:
        for i in labels:
            temp = tag_dic[i]
            taglist[temp] = 1
    tag_label.append(taglist)


print("len alltext=", len(alltext))#未去重前， 5682条
print("len tag_label=", len(tag_label))
alltext_set = set(alltext)#去重后5529条
print(len(alltext_set))# 确实有些句子重复。那么检查是否重复句子的元素识别结果是相同的？


# 预处理。分词，然后构建array。去除停止词吗？？？
tiles = ["sentence"]
for t in tagname_dic:
    tiles.append(tagname_dic[t])

# 将上述的句子和labels进行拼接
label_array = np.array(tag_label)
orign_text_array = np.array(alltext)#不带分词
orign_text_array = orign_text_array.reshape(orign_text_array.shape[0], -1)
print(orign_text_array.shape)
print(label_array.shape)
orign_whole_data_array = np.concatenate([orign_text_array, label_array], axis=1)
orign_whole_data = pd.DataFrame(orign_whole_data_array, columns=tiles)


orign_labor_csv_filename = os.path.join(data_dir, "whole_raw_data_aug.csv")
orign_whole_data.to_csv(orign_labor_csv_filename, sep="\t", encoding="utf-8", index_label="id")#index=0,则不保留行索引,header=0则不保存列名

train_data, test_data = train_test_split(orign_whole_data, random_state=42, test_size=0.1, shuffle=True)
train_path = os.path.join(data_dir, "train_aug.csv")
test_path = os.path.join(data_dir, "dev_aug.csv")
train_data.to_csv(train_path, sep="\t", encoding="utf-8", index_label="id")
test_data.to_csv(test_path, sep="\t", encoding="utf-8", index_label="id")

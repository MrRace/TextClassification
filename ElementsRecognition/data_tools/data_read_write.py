# -*- coding: utf-8 -*-
# @CreatTime    : 2019/9/4 9:52
# @Author  : JasonLiu
# @FileName: data_read_write.py
import json
import collections
import numpy as np
import jieba
import pandas as pd
import data_process_config as cfig
from data_analysis_process import line_process


def get_labels_dict(label_path):
    f = open(label_path, 'r', encoding='utf8')
    tag_dic = {}
    tag_name_dic = {}
    line = f.readline()
    while line:
        tag_name_dic[len(tag_dic)] = line.strip()
        tag_dic[line.strip()] = len(tag_dic)
        line = f.readline()
    f.close()
    return tag_dic, tag_name_dic


def raw_data_to_np(file_path, tag_dic):
    print("file_path=", file_path)
    fin = open(file_path, 'r', encoding='utf8')
    line = fin.readline()
    sentence_tag_dict = collections.OrderedDict()
    while line:
        d = json.loads(line)
        for sent in d:
            sentence = sent['sentence']#.strip()
            labels = sent['labels']  # LB1 - LB20
            if sentence in sentence_tag_dict:
                if sentence_tag_dict[sentence] != labels:
                    # print("same sentence={},label_pre={},label_now={}".format(sentence, sentence_tag_dict[sentence], labels))
                    # 对于不一致的情况，选择第一个labels有标注结果的即可。已有的是空，而新的非空，则覆盖。已有的非空，则不替换。
                    if not sentence_tag_dict[sentence]:
                        sentence_tag_dict[sentence] = labels
            else:
                sentence_tag_dict[sent['sentence']] = labels
        line = fin.readline()
    fin.close()
    print("sentence_tag_dict len=", len(sentence_tag_dict))
    # 遍历sentence_tag_dict
    alltext = []
    tag_label = []
    processed_max_len = 0
    for sentence in sentence_tag_dict:
        # 在此可以加一个文本预处理操作，比如去掉停用词
        if cfig.do_raw_text_process:
            # 进行文本的预处理
            sentence, processed_max_len = line_process(sentence, processed_max_len, is_use_stop_words=False)
        alltext.append(' '.join(jieba.cut(sentence)))
        labels = sentence_tag_dict[sentence]
        taglist = [0] * cfig.num_labels
        if labels:
            for i in labels:
                temp = tag_dic[i]
                taglist[temp] = 1
        tag_label.append(taglist)

    print("processed_max_len=", processed_max_len)  # 预处理后的文本最大长度
    alltext_set = set(alltext)
    if len(alltext_set) != len(sentence_tag_dict):
        print("Maybe exist repeat sentence,please check again!")  # 还是否有重复的呢？
        exit()

    return alltext, tag_label
    # orign_whole_data = pd.DataFrame(orign_whole_data_array, columns=titles)
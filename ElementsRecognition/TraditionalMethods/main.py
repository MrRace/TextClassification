# -*- coding: utf-8 -*-
# @CreatTime    : 2019/9/4 9:48
# @Author  : JasonLiu
# @FileName: main.py
"""
采用传统方法来做多标签文本分类，包括以下方法
(1)SVM
(2)Xgboost
(3)MLP
"""
import numpy as np
import pandas as pd
import os
from myeval import compute_f1
from sklearn.model_selection import train_test_split
from multi_label_models import clf_name, clf_type
from multi_label_models import svm_train_predict
from multi_label_models import xgboost_train_predict
from multi_label_models import mlp_train_predict
from data_tools.data_read_write import get_labels_dict
from data_tools.data_read_write import raw_data_to_np
import warnings
warnings.filterwarnings('ignore')


task_name = "labor"
data_dir = "../data/{}".format(task_name)
train_data_filename = "data_small_selected.json"
raw_file_path = os.path.join(data_dir, train_data_filename)

tag_file = '../data/labor/tags.txt'
tag_dic, tag_name_dic = get_labels_dict(tag_file)
tiles = ["sentence"]
for t in tag_name_dic:
    tiles.append(tag_name_dic[t])

text_list, label_list = raw_data_to_np(raw_file_path, tag_dic)
# 将上述的句子和labels进行拼接
label_array = np.array(label_list)
orign_text_array = np.array(text_list)  # 不带分词
orign_text_array = orign_text_array.reshape(orign_text_array.shape[0], -1)
raw_whole_data_array = np.concatenate([orign_text_array, label_array], axis=1)
whole_data = pd.DataFrame(raw_whole_data_array, columns=tiles)
print("whole_data shape=", whole_data.shape)
train, test = train_test_split(whole_data, random_state=42, test_size=0.1, shuffle=True)
x_train, x_test, y_train, y_test = train_test_split(text_list, label_list,
                                                    random_state=42, test_size=0.1, shuffle=True)

print("text_list len=", len(text_list))
print("label_list len=", len(label_list))
print("x_train len=", len(x_train))
print("x_test len=", len(x_test))
categories = tiles[1:]
prefix_name = "{}_{}".format(task_name, clf_name)

if clf_type == "svm":
    prediction_list, test_list = svm_train_predict(categories, train, test, task_name)
    prediction_array = np.array(prediction_list)
    print(prediction_array.shape)
    prediction_array = prediction_array.T  # 行方向为样本，列方向为类别
    print(prediction_array.shape)
    # 与test的array对比
    test_array = np.array(test_list)
    test_array = test_array.T
    print(test_array.shape)
elif clf_type == "xgboost":
    prediction_array = xgboost_train_predict(x_train, y_train, x_test, task_name)
    test_list = y_test
    test_array = np.array(y_test)
    pass
elif clf_type == "mlp":
    # 可以直接多标签分类
    prediction_array = mlp_train_predict(x_train, y_train, x_test, task_name)  # 注意，y_train是字符型还是数值型
    test_list = y_test
    test_array = np.array(y_test)

"""
之间对预测结果与真实值进行评估.
数据本身类别偏差，即使全部预测为0，准确率也会很高，所以不能用准确率来评估
"""
score_labor = compute_f1(prediction_array, test_array, tag_dic, tag_name_dic)
print('score_labor', score_labor)

# -*- coding: utf-8 -*-
# @CreatTime    : 2019/8/19 18:02
# @Author  : JasonLiu
# @FileName: eval.py

import numpy as np
import pdb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
task_cnt = 20


def gen_new_result(result, truth, label, tag_dic):
    s1 = set()#数字化的标签结果
    for tag in label:
        s1.add(tag_dic[tag.replace(' ', '')])
    s2 = set()
    for name in truth:
        s2.add(tag_dic[name.replace(' ', '')])
    for a in range(0, task_cnt):
        in1 = (a + 1) in s1#是否存在set中,存在的话，值为true
        in2 = (a + 1) in s2
        if in1:
            if in2:#同时存在
                result[0][a]["TP"] += 1
            else:
                result[0][a]["FP"] += 1
        else:
            if in2:
                result[0][a]["FN"] += 1
            else:
                result[0][a]["TN"] += 1
    return result


def get_value(res):
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def gen_score(arr):
    sumf = 0
    y = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for x in arr[0]:
        p, r, f = get_value(x)
        sumf += f
        for z in x.keys():
            y[z] += x[z]

    _, __, f_ = get_value(y)
    return (f_ + sumf * 1.0 / len(arr[0])) / 2.0


def compute_f1(prediction_array, truth_array, tag_dic, tagname_dic):
    row, col = prediction_array.shape
    result = [[]]
    for _ in range(0, task_cnt):  # 每一类都构建一个对应的TP,FP，TN和FN
        result[0].append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})

    # 逐条数据进行统计
    cnt = 0
    for r in range(row):
        prediction_result = prediction_array[r, :]
        truth_result = truth_array[r, :]
        ground_truth = []
        user_output = []
        # pdb.set_trace()
        temp = np.where(prediction_result == 1)[0]  # 返回的是index
        if len(temp) > 0:
            # 获得labels
            for i in temp:
                # print(i)
                ground_truth.append(tagname_dic[i])
        temp1 = np.where(truth_result == 1)[0]
        if len(temp1) > 0:
            for i in temp1:
                user_output.append(tagname_dic[i])
        cnt += 1
        # print("ground_truth=", ground_truth)
        # print("predict=", user_output)
        result = gen_new_result(result, ground_truth, user_output, tag_dic)
    score_labor = gen_score(result)
    return score_labor


def grid_search(train_x, train_y, test_x, test_y, genres, parameters, pipeline):
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    grid_search_tune.fit(train_x, train_y)

    print("Best parameters set:")
    # print(grid_search_tune.best_estimator_.steps)
    print(grid_search_tune.best_params_)

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)

    print(classification_report(test_y, predictions, target_names=genres))
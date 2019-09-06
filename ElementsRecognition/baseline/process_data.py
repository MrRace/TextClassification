# -*- coding: utf-8 -*-
# @CreatTime    : 2019/6/18 16:21
# @Author  : JasonLiu
# @FileName: process_data.py
"""
将数据转为：句子 分类结果
"""
import svm
import json
import numpy as np
import pandas as pd
import jieba
from myeval import compute_f1
from myeval import grid_search
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

filename = ""
filename = '../data/labor/data_small_selected.json'
tag_file = '../data/labor/tags.txt'
fin = open(filename, 'r', encoding='utf8')


def get_label_id(my_dic, q):
    return my_dic[q]


tag_dic, tagname_dic = svm.init(tag_file)
task_cnt = 20
alltext = []
tag_label = []
line = fin.readline()
sentence_tag_dict = {}
while line:
    d = json.loads(line)
    for sent in d:
        alltext.append(sent['sentence'])
        labels = sent['labels']
        taglist = [0] * 20
        if labels:
            for i in labels:
                temp = tag_dic[i]
                taglist[temp] = 1
        tag_label.append(taglist)

        if sent['sentence'] in sentence_tag_dict:
            # print("same sentence={},label_pre={},label_now={}".format(sent['sentence'],
            #                                                           sentence_tag_dict[sent['sentence']], labels))
            # 确实存在少量不一致的情况。初步怀疑是不同人之间的标注差异。有些人认为该句子存在要素，有的人则认为不存在要素
            if sentence_tag_dict[sent['sentence']] != labels:
                print("same sentence={},label_pre={},label_now={}".format(sent['sentence'],
                                                                          sentence_tag_dict[sent['sentence']], labels))
        else:
            sentence_tag_dict[sent['sentence']] = labels
    line = fin.readline()
fin.close()
print('sentence_tag_dict len=', len(sentence_tag_dict))
print("len alltext=", len(alltext))
print("len tag_label=", len(tag_label))
alltext_set = set(alltext)
print(len(alltext_set))# 确实有些句子重复。那么检查是否重复句子的元素识别结果是相同的？

count = 0
train_text = []
for text in alltext:
    count += 1
    if count % 2000 == 0:
        print(count)
    train_text.append(' '.join(jieba.cut(text)))


alltext_array = np.array(train_text)
tag_label_array = np.array(tag_label)
tiles = ["sentence"]
for t in tagname_dic:
    tiles.append(tagname_dic[t])

print(alltext_array.shape)
alltext_array = alltext_array.reshape(alltext_array.shape[0], -1)
print(alltext_array.shape)
print(tag_label_array.shape)
whole_data_array = np.concatenate([alltext_array, tag_label_array], axis=1)
whole_data = pd.DataFrame(whole_data_array, columns=tiles)
print("whole_data shape=", whole_data.shape)
train, test = train_test_split(whole_data, random_state=42, test_size=0.3, shuffle=True)
# NB_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer()),
#                 ('clf', OneVsRestClassifier(MultinomialNB(
#                     fit_prior=True, class_prior=None))),
#             ]) # 结果很差
#

# NB_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer()),
#                 ('clf', OneVsRestClassifier(LinearSVC())),
#             ])#结果大概0.47


# NB_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer()),
#                 ('clf', OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='elasticnet',
#                                                           alpha=1e-4, max_iter=5, random_state=42,
#                                                           shuffle=True, n_jobs=-1))),
#             ])#结果大概0.51

# NB_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer()),
#                 ('clf', OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='elasticnet',
#                                                           alpha=1e-4, max_iter=20, random_state=42,
#                                                           shuffle=True, n_jobs=-1))),
#             ])#结果大概0.52

NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', OneVsOneClassifier(LinearSVC())),
            ])#结果

# NB_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer()),
#                 ('clf', MultinomialNB()),
#             ])#结果大概0.075

parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}

# NB_pipeline = Pipeline([
#                         ('tfidf', TfidfVectorizer()),
#                         ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'))),
#                         ])# 大概是22.5%的指标


categories = tiles[1:]
X_train = train.sentence
X_test = test.sentence

"""
数据本身类别偏差，即使全部预测为0，准确率也会很高，所以不能用准确率来评估

"""
prediction_list = []
test_list = []
for category in categories:
    print('... Processing {}'.format(category))
    # 用 X_dtm & y训练模型
    # print(train[category])
    temp_targets = train[category]
    temp_list = temp_targets.tolist()
    if len(np.unique(temp_targets)) < 2:
        print("only one class")
        continue
    NB_pipeline.fit(X_train, train[category])
    # 计算测试准确率
    prediction = NB_pipeline.predict(X_test)
    prediction_int = prediction.astype(np.int32)
    # prediction_array = np.array(prediction, dtype=int32)
    prediction_index = np.where(prediction_int == 1)#被分为该类别的样本索引
    if len(prediction_index[0]) > 0:
        print("exist")
        print(prediction_index[0])
    print(prediction)#大部分都是0，只有少数位置会是1
    temp = test[category]
    print(temp.values)
    test_list.append(temp.values)
    prediction_list.append(prediction)
    #不等于0，且为1的位置
    # prediction_array = np.concatenate([prediction_array, prediction], axis=1)

prediction_array = np.array(prediction_list)
print(prediction_array.shape)
prediction_array = prediction_array.T#行方向为样本，列方向为类别
print(prediction_array.shape)
    # print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    # print('NB f1 measurement is {} '.format(f1_score(test[category], prediction, average='micro')))
    # sklearn.metrics.classification_report()
# 与test的array对比
test_array = np.array(test_list)
test_array = test_array.T
print(test_array.shape)
joblib.dump(NB_pipeline, 'predictor/model_labor/NB_pipeline.model')

train_x = X_train
y_train = train.iloc[:, 1:].values
train_x = train_x.values.reshape(y_train.shape[0], -1)
print(train_x.shape)
print(y_train.shape)
test_x = X_test
test_x = test_x.values.reshape(test_x.shape[0], -1)
y_test = test.iloc[:, 1:].values
print(test_x.shape)
print(y_test.shape)
# 转为二维list
train_x = train_x.tolist()
y_train = y_train.tolist()
test_x = test_x.tolist()
y_test = y_test.tolist()


# grid_search(train_x, y_train, test_x, y_test, categories, parameters, NB_pipeline)
# joblib.dump(tag, 'predictor/model_labor/tag.model')

# 将testset的预测结果按照真实文件的格式写到文件
# prediction_path = "../output/labor_testdata_prediction.json"
# prediction_file = open(prediction_path, 'r', encoding="utf-8")
# row, col = prediction_array.shape
# for r in range(row):
#     # linedata =
#     predict_data = prediction_array[r, :]
#     # print(predict_data)
#     test_line_data = test_array[r, :]
#     if len(np.where(predict_data == "1")[0]) > 0:
#         if (predict_data == test_line_data).all():
#             print("TP")


"""
之间对预测结果与真实值进行评估
"""

score_labor = compute_f1(prediction_array, test_array, tag_dic, tagname_dic)
print('score_labor', score_labor)


# NB_pipeline.fit(X_train, train[categories])
# prediction = NB_pipeline.predict(X_test)
# print('NB test accuracy is {} '.format(accuracy_score(test[categories], prediction)))
# print('NB f1 measurement is {} '.format(f1_score(test[categories], prediction, average='micro')))
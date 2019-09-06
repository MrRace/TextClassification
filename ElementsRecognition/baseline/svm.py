#!/usr/bin/env python
# coding: utf-8
import json
import numpy as np
import jieba
import os
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


# dim = 5000
dim = 20000


def cut_text(alltext):
    count = 0
    cut = jieba
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append(' '.join(cut.cut(text)))
    return train_text


def train_tfidf(train_data):
    tfidf = TFIDF(
        min_df=9,# 默认值为5
        max_features=dim,# 默认值为5000
        ngram_range=(1, 3),# 默认值为(1, 3)
        use_idf=1,
        smooth_idf=1
    )
    tfidf.fit(train_data)

    return tfidf


def read_trainData(path, tag_path):
    fin = open(path, 'r', encoding='utf8')
    tag_dic, tagname_dic = init(tag_path)
    # tag_dic是LB3-2
    alltext = []
    tag_label = []

    line = fin.readline()
    while line:
        d = json.loads(line)
        for sent in d:
            alltext.append(sent['sentence'])
            tag_label.append(getlabel(sent, tag_dic, True))
        line = fin.readline()
    fin.close()

    return alltext, tag_label


def train_SVC(vec, label):
    # clf = SVC(kernel='rbf', verbose=True)
    # clf.fit(vec, label)
    # return clf
    SVC = LinearSVC()
    SVC.fit(vec, label)
    return SVC


def conver_labels_array(datas_label):
    """
    将tag_labe转为矩阵
    :param tag_label:
    :return:
    """
    # if not is_multi_labels:
    #     # 单label
    labels_array = []
    for i in range(len(datas_label)):
        labels = datas_label[i]
        taglist = [0] * 20
        if labels:
            # 非空
            for j in labels:
                taglist[j] = 1#相应位置置为1
        labels_array.append(taglist)
    labels_array = np.array(labels_array)
    return labels_array


def binary_relevance(vec, label):
    """
    :param vec: 矩阵形式的X，此时的X已经是向量化的数字
    :param label: 矩阵形式的Y
    :return:
    """
    from skmultilearn.problem_transform import BinaryRelevance
    from skmultilearn.problem_transform import ClassifierChain
    from xgboost import XGBClassifier
    from skmultilearn.problem_transform import LabelPowerset
    from skmultilearn.adapt import MLkNN
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    # 用一个基于高斯朴素贝叶斯的分类器
    # classifier = BinaryRelevance(GaussianNB())# 初始化二元关联多标签分类器
    # classifier = ClassifierChain(GaussianNB(), require_dense=[False, True])
    # classifier = BinaryRelevance(GaussianNB())
    # classifier = LabelPowerset(GaussianNB())#标签Powerset（Label Powerset）
    # classifier = LabelPowerset(LinearSVC())#标签Powerset（Label Powerset#离线是98.35。目前线上最优，但是只有58%。所以，貌似是过拟合了
    # classifier = MLkNN(k=20)#55.86
    # classifier = LabelPowerset(XGBClassifier(n_jobs=-1, max_depth=6))#离线87.97%，线上测试只有52.306%。同样过拟合了。或者直接采用更大的数据集测试？？？
    # classifier = DecisionTreeClassifier()# 使用决策树进行多标签分类。离线是99.26，在线是49.445%
    classifier = RandomForestClassifier(random_state=1)# 使用随机森林进行多标签分类,离线指标是92.54%，线上测试是47.337%
    x_train, y_train = vec, label
    classifier.fit(x_train, y_train)# 直接预测数字？
    return classifier


def single_classical_models(vec, label):
    """
    主要采用经典的单模型解决多标签分类问题
    :return:
    """
    from sklearn.neural_network import MLPClassifier
    model_name = "mlp"
    # 用一个基于高斯朴素贝叶斯的分类器
    if model_name == "mlp":
        classifier = MLPClassifier()#离线测试99.3%，在线测试指标为54.896%。在分为训练集和校验集后，更为妥当。在校验集上指标为77%
    elif model_name == "svc":
        from skmultilearn.problem_transform import LabelPowerset
        classifier = LabelPowerset(LinearSVC())
    # 所以，可以先在校验集调参后，在应用于整个数据集的训练，再到线上预测.
    # 即使调参后，线上指标也仅有55.153%
    x_train, y_train = vec, label
    classifier.fit(x_train, y_train)  # 直接预测数字？
    return classifier


def emsemble_basic_classifiers(vec, label):
    """
    采用集成学习的方式。已经在线上测试过
    :return:
    """
    from skmultilearn.problem_transform import BinaryRelevance
    from skmultilearn.problem_transform import ClassifierChain
    from xgboost import XGBClassifier
    from skmultilearn.problem_transform import LabelPowerset
    from skmultilearn.adapt import MLkNN
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import VotingClassifier
    # classifier_svc = LabelPowerset(LinearSVC())
    classifier_deci = DecisionTreeClassifier(random_state=666)
    classifier_rf = RandomForestClassifier(random_state=1)
    basic_classifiers = [('svm_clf', LabelPowerset(LinearSVC())),
                         ('dec_clf', classifier_deci),
                         ('rf_clf', classifier_rf)
                         ]
    # voting_clf = VotingClassifier(estimators=basic_classifiers, voting='soft')
    # voting_clf = VotingClassifier(estimators=basic_classifiers, voting='hard')

    # classifier = RandomForestClassifier(n_estimators=10)#离线指标93.38，所谓的离线都是直接面向训练集的评估。后续再划分为训练集和开发集
    classifier = ExtraTreesClassifier(n_estimators=3) #极大随机树方法，离线指标99.26%,在线评估是46.025%。这些显然都是因为过拟合。即使是修改了n_estimators参数为10，5，3都是没啥变化

    x_train, y_train = vec, label
    classifier.fit(x_train, y_train)  # 直接预测数字？
    return classifier


def emsemble_bagging(vec, label):
    """
    采用集成学习的方式-bagging
    :return:
    """
    from sklearn.ensemble import BaggingClassifier
    from skmultilearn.problem_transform import LabelPowerset
    from skmultilearn.adapt import MLkNN
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import VotingClassifier
    # classifier_svc = LabelPowerset(LinearSVC())
    classifier_deci = DecisionTreeClassifier(random_state=666)
    classifier_rf = RandomForestClassifier(random_state=1)
    basic_classifiers = [('svm_clf', LabelPowerset(LinearSVC())),
                         ('dec_clf', classifier_deci),
                         ('rf_clf', classifier_rf)
                         ]
    # voting_clf = VotingClassifier(estimators=basic_classifiers, voting='soft')
    # voting_clf = VotingClassifier(estimators=basic_classifiers, voting='hard')

    # classifier = RandomForestClassifier(n_estimators=10)#离线指标93.38，所谓的离线都是直接面向训练集的评估。后续再划分为训练集和开发集
    classifier = ExtraTreesClassifier(n_estimators=3) #极大随机树方法，离线指标99.26%,在线评估是46.025%。这些显然都是因为过拟合。即使是修改了n_estimators参数为10，5，3都是没啥变化

    x_train, y_train = vec, label
    classifier.fit(x_train, y_train)  # 直接预测数字？
    return classifier


def emsemble_boosting(vec, label):
    """
    采用集成学习的方式-boosting
    :return:
    """


def emsemble_stacking(vec, label):
    """
    采用集成学习的方式-stacking
    :return:
    """


def xgboost_sklearn():
    from xgboost import XGBClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    pass


def init(tags_path):
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


def getlabel(d, tag_dic, is_get_multilabels=False):
    # 做单标签
    # 返回多个类的第一个。这个处理比较粗糙
    if len(d['labels']) > 0:
        # print(d['labels'])
        if not is_get_multilabels:
            return tag_dic[d['labels'][0]]
        else:
            labels = []
            for i in range(len(d['labels'])):
                labels.append(tag_dic[d['labels'][i]])
            return labels
    return ''


if __name__ == '__main__':
    stage_num = 2
    print('train_labor_model...')
    print('reading...')
    if stage_num == 1:
        labor_data_path = "../data/labor/"
        labor_data_filename = "data_small_selected.json"
    else:
        labor_data_path = "../data/CAIL2019-FE-big/labor"
        labor_data_filename = "train_selected.json"
    labor_data_file = os.path.join(labor_data_path, labor_data_filename)
    labor_tag_file = os.path.join(labor_data_path, "tags.txt")

    alltext, tag_label = read_trainData(labor_data_file, labor_tag_file)# list格式，每个成员为一个句子。tag_label是每个句子对应的label
    print('cut text...')
    train_data = cut_text(alltext)#返回分词的结果，每个成员是一个句子。句子中用空格将word分隔开
    print('train tfidf...')
    tfidf = train_tfidf(train_data)

    vec = tfidf.transform(train_data)#根据找到的规则对数据进行转换

    print('tag SVC')
    # tag = train_SVC(vec, tag_label)
    tag_array = conver_labels_array(tag_label)
    # tag = binary_relevance(vec, tag_array) # 单模型
    tag = single_classical_models(vec, tag_array)  # 集成学习模型

    print('saving model')
    joblib.dump(tfidf, 'predictor/model_labor/tfidf.model')
    joblib.dump(tag, 'predictor/model_labor/tag.model')


    print('train_divorce_model...')
    print('reading...')
    if stage_num == 1:
        divorce_data_path = "../data/divorce/"
        divorce_data_filename = "data_small_selected.json"
    else:
        divorce_data_path = "../data/CAIL2019-FE-big/divorce"
        divorce_data_filename = "train_selected.json"
    divorce_data_file = os.path.join(divorce_data_path, divorce_data_filename)
    divorce_tag_file = os.path.join(divorce_data_path, "tags.txt")
    alltext, tag_label = read_trainData(divorce_data_file, divorce_tag_file)
    print('cut text...')
    train_data = cut_text(alltext)
    print('train tfidf...')
    tfidf = train_tfidf(train_data)
    vec = tfidf.transform(train_data)

    print('tag SVC')
    # tag = train_SVC(vec, tag_label)
    tag_array = conver_labels_array(tag_label)
    # tag = binary_relevance(vec, tag_array)# 单模型
    tag = single_classical_models(vec, tag_array) # 集成学习模型

    print('saving model')
    joblib.dump(tfidf, 'predictor/model_divorce/tfidf.model')
    joblib.dump(tag, 'predictor/model_divorce/tag.model')

    print('train_loan_model...')
    print('reading...')
    if stage_num == 1:
        loan_data_path = "../data/loan/"
        loan_data_filename = "data_small_selected.json"
    else:
        loan_data_path = "../data/CAIL2019-FE-big/loan"
        loan_data_filename = "train_selected.json"
    loan_data_file = os.path.join(loan_data_path, loan_data_filename)
    loan_tag_file = os.path.join(loan_data_path, "tags.txt")
    alltext, tag_label = read_trainData(loan_data_file, loan_tag_file)
    print('cut text...')
    train_data = cut_text(alltext)
    print('train tfidf...')
    tfidf = train_tfidf(train_data)
    vec = tfidf.transform(train_data)

    print('tag SVC')
    # tag = train_SVC(vec, tag_label)
    tag_array = conver_labels_array(tag_label)
    # tag = binary_relevance(vec, tag_array)
    tag = single_classical_models(vec, tag_array)  # 集成学习模型

    print('saving model')
    joblib.dump(tfidf, 'predictor/model_loan/tfidf.model')
    joblib.dump(tag, 'predictor/model_loan/tag.model')

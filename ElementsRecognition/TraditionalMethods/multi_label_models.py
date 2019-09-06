# -*- coding: utf-8 -*-
# @CreatTime    : 2019/9/4 9:50
# @Author  : JasonLiu
# @FileName: svm.py
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

clf_type = "xgboost"  # 支持svm  xgboost   mlp严格上来说，划分得不细
clf_name = "one_vs_rest_xgboost"
is_do_grid_search = True  # 是否进行超参数搜索

clf_dict = {
    "one_vs_rest_multinb": OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None)),
    "one_vs_rest_svm": OneVsRestClassifier(LinearSVC()),
    "one_vs_rest_sgd": OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=1e-4,
                                           max_iter=5, random_state=42, shuffle=True, n_jobs=-1)),  # 超参应该调试
    "one_vs_one_svm": OneVsOneClassifier(LinearSVC()),
    "one_vs_rest_xgboost": OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=6)), #有问题
    "mulnb": MultinomialNB(),  # 结果一般是很差的,大概0.075
    "mlp":  MLPClassifier()  #
}
# onev_v_one_mulNB_pipeline = Pipeline([
#                                     ('tfidf', TfidfVectorizer()),
#                                     ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
#                                     ])
#
# onev_v_one_svm_pipeline = Pipeline([
#                                     ('tfidf', TfidfVectorizer()),
#                                     ('clf', OneVsRestClassifier(LinearSVC())),
#                                    ])#结果大概0.47

#
# NB_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer()),
#                 ('clf', OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='elasticnet',
#                                                           alpha=1e-4, max_iter=5, random_state=42,
#                                                           shuffle=True, n_jobs=-1))),
#             ])#结果大概0.51

# NB_pipeline = Pipeline([
#                         ('tfidf', TfidfVectorizer()),
#                         ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'))),
#                         ])# 大概是22.5%的指标

multi_labels_pipeline = Pipeline([
                                ('tfidf', TfidfVectorizer()),
                                ('clf',  clf_dict[clf_name]),
                                ])#结果


parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}


def save_model(task_name):
    # 保存训练后的模型
    models_dir = "./models/model_{}".format(task_name)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)  #
    model_name = "{}.model".format(clf_name)
    models_path = os.path.join(models_dir, model_name)
    joblib.dump(multi_labels_pipeline, models_path)


def svm_train_predict(categories, train, test, task_name):
    """
    采用svm进行多标签文本分类
    :param categories:
    :param train:
    :param test:
    :param task_name:
    :return:
    """
    prediction_list = []
    test_list = []
    X_train = train.sentence
    X_test = test.sentence
    for category in categories:
        print('Processing {}'.format(category))
        # 用 X_dtm & y训练模型
        # print(train[category])
        temp_targets = train[category]
        if len(np.unique(temp_targets)) < 2:
            print("only one class")  # 即该类别目前没有样本，全部是0
            continue
        # prediction.astype(np.int32)
        multi_labels_pipeline.fit(X_train, train[category].astype(np.int32))
        # 计算测试准确率
        prediction = multi_labels_pipeline.predict(X_test)
        temp = test[category].astype(np.int32)
        test_list.append(temp.values)
        prediction_list.append(prediction)
        save_model(task_name)
    return prediction_list, test_list


def xgboost_train_predict(x_train, y_train, x_test, task_name):
    y_train = np.array(y_train)
    y_train = y_train.astype(np.int32)
    if is_do_grid_search:
        param_grid = {
            'clf__estimator__scale_pos_weight': [5, 10, 15],  # 当正负样本比例为1:10时，scale_pos_weight=10。
            "clf__estimator__max_depth": [2, 4, 6, 8, 10],  # 树的深度，默认值为6，典型值3-10
            "clf__estimator__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。
            'clf__estimator__n_estimators': [500, 1000, 2000, 3000],  # 总共迭代的次数，即决策树的个数
            'clf__estimator__learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
        }
        do_grid_search(multi_labels_pipeline, param_grid, x_train, y_train)
    else:
        multi_labels_pipeline.fit(x_train, y_train)
        prediction = multi_labels_pipeline.predict(x_test)
        save_model(task_name)
    return prediction


def mlp_train_predict(x_train, y_train, x_test, task_name):
    """
    采用MLP进行多标签文本分类
    :param x_train:
    :param y_train:
    :param x_test:
    :param task_name:
    :return:
    """
    # 构建x_train, y_train
    if is_do_grid_search:
        param_grid = {'tfidf__max_features': [5000, 10000, 15000, 20000],
                      'tfidf__min_df': (4, 5, 6, 7, 8, 9, 10, 11, 12),
                      'tfidf__ngram_range': [(1, 3), (1, 4), (1, 5)]  # 可以加其他需要调节的超参数
                      }
        do_grid_search(multi_labels_pipeline, param_grid, x_train, y_train)
    else:
        multi_labels_pipeline.fit(x_train, y_train)
        prediction = multi_labels_pipeline.predict(x_test)  # 此时csr_matrix类型
        save_model(task_name)
    return prediction


def do_grid_search(pipe, param_grid, data_x, data_y):
    classifier = GridSearchCV(pipe, param_grid, cv=2, n_jobs=4, verbose=3)
    classifier.fit(data_x, data_y)  # 此时的train_text是list形式
    print("Best Score: ", classifier.best_score_)
    print("Best Params: ", classifier.best_params_)

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
import scipy
from myeval import compute_f1
from numpy import mat
from myeval import grid_search
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.multiclass import OneVsRestClassifier
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

dim = 5000


PhaseNum = 1# 第1阶段，数据量较少
if PhaseNum == 1:
    filename = '../data/labor/data_small_selected.json'
    tag_file = '../data/labor/tags.txt'
else:
    filename = '../data/CAIL2019-FE-big/labor/train_selected.json'
    tag_file = '../data/CAIL2019-FE-big/labor/tags.txt'
fin = open(filename, 'r', encoding='utf8')


def get_label_id(my_dic, q):
    return my_dic[q]


tag_dic, tagname_dic = svm.init(tag_file)#构建映射词典，包括labe-id，id-label。这里id从0开始。而Label从LB1开始
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
            # print("same sentence={},label_pre={},label_now={}".format(sentence,
            #                                                           sentence_tag_dict[sentence], labels))
            # 确实存在少量不一致的情况。初步怀疑是不同人之间的标注差异。有些人认为该句子存在要素，有的人则认为不存在要素
            if sentence_tag_dict[sentence] != labels:
                print("same sentence={},label_pre={},label_now={}".format(sentence,
                                                                          sentence_tag_dict[sentence], labels))
                # 对于不一致的情况，选择第一个labels有标注结果的即可。已有的是空，而新的非空，则覆盖。已有的非空，则不替换。
                if not sentence_tag_dict[sentence]:
                    sentence_tag_dict[sentence] = labels
                    print("wap,sent=", sentence)
        else:
            sentence_tag_dict[sent['sentence']] = labels
    line = fin.readline()
fin.close()
# 遍历sentence_tag_dict
alltext = []
tag_label = []
for sentence in sentence_tag_dict:
    alltext.append(sentence)
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
count = 0
train_text = []
for text in alltext:
    count += 1
    if count % 2000 == 0:
        print(count)
    train_text.append(' '.join(jieba.cut(text)))

tfidf = TfidfVectorizer(
                        min_df=9,
                        max_features=dim,
                        ngram_range=(1, 3),
                        use_idf=1,
                        smooth_idf=1
                        )
tfidf.fit(train_text)#此时的train_text是list形式
#以tf-idf将文本向量化
vec = tfidf.transform(train_text)#这个即为所有word的向量

alltext_array = np.array(train_text)
tag_label_array = np.array(tag_label)

print(alltext_array.shape)
alltext_array = alltext_array.reshape(alltext_array.shape[0], -1)
print(alltext_array.shape)
print(tag_label_array.shape)
whole_data_array = np.concatenate([alltext_array, tag_label_array], axis=1)
tiles = ["sentence"]
for t in tagname_dic:
    tiles.append(tagname_dic[t])
whole_data = pd.DataFrame(whole_data_array, columns=tiles)
# 到此，完成数据的组装。行方向是样本，列方向是labels
# whole_data.to_csv()


train, test = train_test_split(whole_data, random_state=42, test_size=0.2, shuffle=True)
# train = whole_data

def binary_relevance(train_data, test_data):
    """
    可以正常运行和预测
    使用二元关联。
    仅仅选取一个分类结果，即将问题简化为多分类单标签问题。而实际问题是多分类多标签问题。
    :param train_data:
    :param test_data:
    :return:
    """

    from skmultilearn.problem_transform import BinaryRelevance
    from skmultilearn.problem_transform import ClassifierChain
    from sklearn.naive_bayes import GaussianNB
    # 用一个基于高斯朴素贝叶斯的分类器
    # classifier = BinaryRelevance(GaussianNB())# 初始化二元关联多标签分类器
    classifier = ClassifierChain(GaussianNB())
    #X_train = train
    X_train, y_train = train_data.iloc[:, [0]], train_data.iloc[:, list(range(1, 21))]
    X_test, y_test = test_data.iloc[:, [0]], test_data.iloc[:, list(range(1, 21))]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # 训练
    temp = X_train.values.tolist()
    X = []
    for i in range(len(temp)):
        X.append(temp[i][0])
    x = tfidf.transform(X)
    y = y_train.values.tolist()
    Y = []#长度为20的矩阵
    for j in range(len(y)):
        if "1" in y[j]:
            indexs = y[j].index("1")
            Y.append(indexs+1)
        else:
            # print("0")
            Y.append(0)#其实有21类，因为有空
    Y = np.array(Y)
    # Y值不能是多值？？
    classifier.fit(x, Y)# 直接预测数字？
    """
    报错：raise TypeError('no supported conversion for types: %r' % (args,))
    TypeError: no supported conversion for types: (dtype('O'),)
    难道是？？
    """

    # 预测
    temp = X_test.values.tolist()
    X_ts = []
    for i in range(len(temp)):
        X_ts.append(temp[i][0])
    x_test = tfidf.transform(X_ts)

    y_test = y_test.values.tolist()
    Y_test = []
    for j in range(len(y_test)):
        if "1" in y_test[j]:
            indexs = y_test[j].index("1")
            Y_test.append(indexs + 1)
        else:
            # print("0")
            Y_test.append(0)  # 其实有21类，因为有空
    Y_test = np.array(Y_test)#形成一个矩阵
    unique_test, counts_test = np.unique(Y_test, return_counts=True)
    print("truth=", dict(zip(unique_test, counts_test)))

    predictions = classifier.predict(x_test)#此时csr_matrix类型
    predictions = predictions.toarray()
    # 里面有0吗？？
    unique, counts = np.unique(predictions, return_counts=True)
    print("preditions=", dict(zip(unique, counts)))
    from sklearn.metrics import accuracy_score
    score = accuracy_score(Y_test, predictions)
    print(score)


def multil_labels_binary_relevance(train_data, test_data):
    """
    可以正常运行
    使用二元关联。多分类多标签问题。
    :param train_data:
    :param test_data:
    :return:
    """
    from skmultilearn.problem_transform import BinaryRelevance
    from skmultilearn.problem_transform import ClassifierChain
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    xgt_param = {'max_depth': 6, 'eta': 0.5, 'eval_metric': 'merror', 'silent': 1,
                 'objective': 'multi:softmax', 'num_class': 20}
    # 用一个基于高斯朴素贝叶斯的分类器
    classifier = BinaryRelevance(GaussianNB())# 初始化二元关联多标签分类器
    # classifier = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))
    X_train, y_train = train_data.iloc[:, [0]], train_data.iloc[:, list(range(1, 21))]
    X_test, y_test = test_data.iloc[:, [0]], test_data.iloc[:, list(range(1, 21))]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # 训练
    temp = X_train.values.tolist()
    X = []
    for i in range(len(temp)):
        X.append(temp[i][0])
    x = tfidf.transform(X)
    y = y_train.values.tolist()
    # Y = [0]*20#长度为20的矩阵
    # for j in range(len(y)):
    #     if "1" in y[j]:
    #         indexs = y[j].index("1")
    #         Y.append(indexs+1)
    #     else:
    #         # print("0")
    #         Y.append(0)#其实有21类，因为有空
    Y = np.array(y)
    Y = Y.astype(np.int32)

    # Y值不能是多值？？
    classifier.fit(x, Y)# 直接预测数字？
    """
    报错：raise TypeError('no supported conversion for types: %r' % (args,))
    TypeError: no supported conversion for types: (dtype('O'),)
    难道是？？
    """

    # 预测
    temp = X_test.values.tolist()
    X_ts = []
    for i in range(len(temp)):
        X_ts.append(temp[i][0])
    x_test = tfidf.transform(X_ts)

    y_test = y_test.values.tolist()
    # Y_test = []
    # for j in range(len(y_test)):
    #     if "1" in y_test[j]:
    #         indexs = y_test[j].index("1")
    #         Y_test.append(indexs + 1)
    #     else:
    #         # print("0")
    #         Y_test.append(0)  # 其实有21类，因为有空
    Y_test = np.array(y_test)#形成一个矩阵
    Y_test = Y_test.astype(np.int32)
    unique_test, counts_test = np.unique(Y_test, return_counts=True)
    print("truth=", dict(zip(unique_test, counts_test)))

    predictions = classifier.predict(x_test)#此时csr_matrix类型
    predictions = predictions.toarray()
    # 里面有0吗？？
    unique, counts = np.unique(predictions, return_counts=True)
    print("preditions=", dict(zip(unique, counts)))
    from sklearn.metrics import accuracy_score
    score = accuracy_score(Y_test, predictions)
    print(score)


def single_multilabel_model(train_data, test_data, whole_data_x, whole_data_y):
    """
    使用常见的单模型多标签分类器
    :return:
    """
    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier


    is_gridsearch_cv = False
    model_name = "mlp"

    if is_gridsearch_cv:
        pipe = Pipeline([
                          ('tfidf', TfidfVectorizer()),
                          ('clf', MLPClassifier())
                        ])
        param_grid = {'tfidf__max_features': [5000, 10000, 15000, 20000],
                      'tfidf__min_df': (4, 5, 6, 7, 8, 9, 10, 11, 12),
                      'tfidf__ngram_range': [(1, 2), (1, 3), (1, 4), (1, 5)]
                      }
        # Best Score: 0.7703020437692168
        # Best Params: {'tfidf__ngram_range': (1, 3), 'tfidf__max_features': 20000, 'tfidf__min_df': 9}

        # param_grid = {'tfidf__min_df': (2, 3, 4, 5, 6, 7, 8),
        #               'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)]
        #               }#Best Params:  {'tfidf__min_df': 8, 'tfidf__ngram_range': (1, 3)}

        classifier = GridSearchCV(pipe, param_grid, cv=2, n_jobs=4, verbose=3)
        classifier.fit(whole_data_x, whole_data_y)  # 此时的train_text是list形式
        # classifier.fit()
        print("Best Score: ", classifier.best_score_)
        print("Best Params: ", classifier.best_params_)
    else:
        # 单模型
        if model_name == "mlp":
            classifier = MLPClassifier()#alpha=0.7，其最后的指标得分为75.226%；而使用默认参数，指标为78.84%
        elif model_name == "xgboost":
            pass
        elif model_name == "svm":
            pass
        elif model_name == "lightgbm":
            pass
        elif model_name == "RandomForest":
            pass
        elif model_name == "":
            pass

        X_train, y_train = train_data.iloc[:, [0]], train_data.iloc[:, list(range(1, 21))]
        X_test, y_test = test_data.iloc[:, [0]], test_data.iloc[:, list(range(1, 21))]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        # 训练
        temp = X_train.values.tolist()
        X = []
        for i in range(len(temp)):
            X.append(temp[i][0])
        x = tfidf.transform(X)
        y = y_train.values.tolist()
        Y = np.array(y)
        Y = Y.astype(np.int32)
        # Y值不能是多值？？
        classifier.fit(x, Y)# 直接预测数字？
        # 预测
        temp = X_test.values.tolist()
        X_ts = []
        for i in range(len(temp)):
            X_ts.append(temp[i][0])
        x_test = tfidf.transform(X_ts)
        y_test = y_test.values.tolist()
        Y_test = np.array(y_test)#形成一个矩阵
        Y_test = Y_test.astype(np.int32)
        unique_test, counts_test = np.unique(Y_test, return_counts=True)
        print("truth=", dict(zip(unique_test, counts_test)))

        predictions = classifier.predict(x_test)#此时csr_matrix类型
        # predictions = predictions.toarray()
        # 里面有0吗？？
        unique, counts = np.unique(predictions, return_counts=True)
        print("preditions=", dict(zip(unique, counts)))
        from sklearn.metrics import accuracy_score
        score = accuracy_score(Y_test, predictions)
        print(score)


def transform_to_matrix(x, w2v_model, vec_size=128, padding_size=256):
    """
    采用word2vec，而不用tf-idf。
    :param x: 待转为向量的原始句子
    :param w2v_model: 预训练的词向量模型
    :param padding_size:
    :param vec_size:
    :return:
    """
    res = []
    for sen in x:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(w2v_model[sen[i]].tolist())
            except:
                # 这里有两种except情况，
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec
                matrix.append([0] * vec_size)# 全 0 合适吗？
        res.append(matrix)
    return res


def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))#向量逐点求和
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def get_whole_law_corpus():
    """
    收集所有法律语料，进行词向量的训练
    :return:
    """
    import jieba
    from tqdm import tqdm

    corpus_text = []
    # 1：阅读理解语料
    mrc_files = ["../../ReadingComprehension/baseline/data/big_train_data.json"]
    for mrc_file in mrc_files:
        with open(mrc_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)["data"]
            for entry in tqdm(input_data):#遍历各个样本
                for paragraph in entry["paragraphs"]:
                    paragraph_text = paragraph["context"]
                    paragraph_text = paragraph_text.strip()
                    corpus_text.append(' '.join(jieba.cut(paragraph_text)))

    # 2：要素识别语料
    er_files = ["../data/CAIL2019-FE-big/divorce/train_selected.json",
                "../data/CAIL2019-FE-big/loan/train_selected.json",
                "../data/CAIL2019-FE-big/labor/train_selected.json"]
    for er_file in er_files:
        with open(er_file, 'r', encoding='utf-8') as f_er:
            for line_data in f_er.readlines():
                line_text = json.loads(line_data)
                for sen in line_text:
                    sentence = sen['sentence']
                    corpus_text.append(' '.join(jieba.cut(sentence)))

    # 3：案情相似语料
    scm_files = ["../../SimilarCaseMatching/data/CAIL2019-SCM-big.json"]
    for scm_file in scm_files:
        with open(scm_file, 'r', encoding='utf-8') as f_scm:
            for line_data in f_scm.readlines():
                line_text = json.loads(line_data)
                a_case = line_text["A"].strip()
                a_case = a_case.replace("\n", "")
                a_case = a_case.replace("\r", "")
                b_case = line_text["B"].strip().replace("\n", "").replace("\r", "")
                c_case = line_text["C"].strip().replace("\n", "").replace("\r", "")
                corpus_text.append(' '.join(jieba.cut(a_case)))
                corpus_text.append(' '.join(jieba.cut(b_case)))
                corpus_text.append(' '.join(jieba.cut(c_case)))

    # 尝试去重
    print("len of corpus_text=", len(corpus_text))
    import os
    import pickle
    from gensim.models import word2vec
    n_dim = 100
    is_overwrite = False  # 是否重新训练词向量，否则直接采用已经训练好的词向量
    law_pretrain_file = os.path.join("../data/", 'w2v_law_whole.pkl')
    is_exist_pretrain_w2v = os.path.exists(law_pretrain_file)
    # 由于语料是固定的，或者说短期固定，可以将分词结果保存到磁盘，直接加载就行
    w2v_model = word2vec.Word2Vec(corpus_text, size=n_dim, min_count=2, workers=12, iter=50)
    # 保存模型
    w2v_model.save(law_pretrain_file)  # 默认是wb
    """
    # 第二种训练方式
    new_model = gensim.models.Word2Vec(min_count=1)  # 先启动一个空模型 an empty model
    new_model.build_vocab(sentences)                 # can be a non-repeatable, 1-pass generator     
    new_model.train(sentences, total_examples=new_model.corpus_count, epochs=new_model.iter)                  
    """

def wordembedding_mlp(x_data, y_data):
    """
    使用词嵌入的方式+多层感知机的方式
    :param x_data: 纯文本的原始数据
    :param y_data: 已经数字化的结果标签
    :return:
    """
    # 加载预训练的词向量。该词向量的训练是基于此次3个比赛的数据集，即阅读理解+元素识别+案情相似
    n_dim = 100
    is_use_pretrain_w2v = False

    if is_use_pretrain_w2v:
        import gensim
        word2vec_file_path = "/home/liujiepeng/LM/pretraining_models/word_embedding/wiki.zh.text.vector"
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file_path, binary=False, unicode_errors='ignore')
        # 如果直接采用中文wiki的词向量，离线dev set上的评估结果为73.23%
    else:
        # 重新训练语料
        import os
        import pickle
        from gensim.models import word2vec

        is_overwrite = False#是否重新训练词向量，否则直接采用已经训练好的词向量
        law_pretrain_file = os.path.join("../data/", 'w2v_law_whole.pkl')
        is_exist_pretrain_w2v = os.path.exists(law_pretrain_file)
        if is_exist_pretrain_w2v and not is_overwrite:
            # 使用已有的词向量
            w2v_model = pickle.load(open(law_pretrain_file, 'rb'))
        else:
            # 不存在 或者 需要覆盖旧的词向量
            # get_whole_law_corpus()# 基于语料的预训练,这是完全独立的一个功能模块

            corpus = x_data# 每个句子是
            corpus = corpus.tolist()
            # 由于语料是固定的，或者说短期固定，可以将分词结果保存到磁盘，直接加载就行
            w2v_model = word2vec.Word2Vec(corpus, size=n_dim, min_count=2, workers=12, iter=10)
            # 保存模型
            w2v_model.save(law_pretrain_file)#默认是wb

    train_data, test_data, y_train, y_test = train_test_split(x_data, y_data, random_state=42, test_size=0.2, shuffle=True)

    # 通过词嵌入将句子转为矩阵向量
    # X_train = transform_to_matrix(train_data, w2v_model, vec_size=400)这个会形成3维矩阵的，第一维是样本数，第二维是每个样本的长度，第3维是每个词的词嵌入长度
    # X_test = transform_to_matrix(test_data, w2v_model, vec_size=400)

    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, w2v_model) for z in train_data])
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, w2v_model) for z in test_data])

    # 转成np的数组，便于处理
    X_train = np.array(train_vecs)
    X_test = np.array(test_vecs)
    # 看看数组的大小
    print(X_train.shape)
    print(X_test.shape)

    # 输入到模型中
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier()
    # 模型训练
    classifier.fit(X_train, y_train)

    # 对训练的模型进行评估
    test_prediction = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, test_prediction)
    print(score)
    """
    不管是采用外界第三方已经训练好的词向量，比如wiki中文或者自己基于法律文书训练的词向量，其最终得分也一直是73.23%，基础维持不变
    且词嵌入的维度从100-400的变化，也对结果不影响。
    """


def multil_labels_sklearn_xgboost(train_data, test_data):
    """
    xgboost。多分类多标签问题。
    :param train_data:
    :param test_data:
    :return:
    """
    from xgboost import XGBClassifier
    from skmultilearn.problem_transform import LabelPowerset
    from sklearn.preprocessing import MultiLabelBinarizer
    xgt_param = {'max_depth': 6, 'eta': 0.5, 'eval_metric': 'merror', 'silent': 1,
                 'objective': 'multi:softmax', 'num_class': 20}

    # classifier = XGBClassifier(learning_rate=0.1,
    #                            n_estimators=1000,  # 树的个数--1000棵树建立xgboost
    #                            max_depth=6,  # 树的深度
    #                            min_child_weight=1,  # 叶子节点最小权重
    #                            gamma=0.,  # 惩罚项中叶子结点个数前的参数
    #                            subsample=0.8,  # 随机选择80%样本建立决策树
    #                            colsample_btree=0.8,  # 随机选择80%特征建立决策树
    #                            objective='multi:softmax',  # 指定损失函数
    #                            scale_pos_weight=1,  # 解决样本个数不平衡的问题
    #                            random_state=27  # 随机数
    #                            )

    # classifier = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=6, scale_pos_weight=1))#最终指标79.26%
    classifier = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=6))# 最终指标79.26
    # classifier = LabelPowerset(XGBClassifier(n_jobs=-1, max_depth=6))# 最终指标
    X_train, y_train = train_data.iloc[:, [0]], train_data.iloc[:, list(range(1, 21))]
    X_test, y_test = test_data.iloc[:, [0]], test_data.iloc[:, list(range(1, 21))]
    # 训练
    temp = X_train.values.tolist()
    X = []
    for i in range(len(temp)):
        X.append(temp[i][0])
    x = tfidf.transform(X)
    y = y_train.values.tolist()
    Y = np.array(y)
    Y = Y.astype(np.int32)

    mlb = MultiLabelBinarizer()
    # Y = mlb.fit_transform(Y)
    classifier.fit(x, Y)# 直接预测数字？

    # 预测
    temp = X_test.values.tolist()
    X_ts = []
    for i in range(len(temp)):
        X_ts.append(temp[i][0])
    x_test = tfidf.transform(X_ts)

    y_test = y_test.values.tolist()
    Y_test = np.array(y_test)#形成一个矩阵
    Y_test = Y_test.astype(np.int32)
    unique_test, counts_test = np.unique(Y_test, return_counts=True)
    print("truth=", dict(zip(unique_test, counts_test)))

    predictions = classifier.predict(x_test)#此时ndarray类型
    # predictions = predictions.toarray()
    # 里面有0吗？？
    unique, counts = np.unique(predictions, return_counts=True)
    print("preditions=", dict(zip(unique, counts)))
    from sklearn.metrics import accuracy_score
    score = accuracy_score(Y_test, predictions)
    print(score)


def multil_labels_xgboost(train_data, test_data):
    """
    使用xgboost。多分类多标签问题。
    :param train_data:
    :param test_data:
    :return:
    """

    # 用一个基于高斯朴素贝叶斯的分类器
    from xgboost import XGBClassifier
    classifier = XGBClassifier(learning_rate=0.1,
                               n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                               max_depth=6,               # 树的深度
                               min_child_weight=1,        # 叶子节点最小权重
                               gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                               subsample=0.8,             # 随机选择80%样本建立决策树
                               colsample_btree=0.8,       # 随机选择80%特征建立决策树
                               objective='multi:softmax', # 指定损失函数
                               scale_pos_weight=1,        # 解决样本个数不平衡的问题
                               random_state=27            # 随机数
                               )
    x_train, y_train = train_data.iloc[:, [0]], train_data.iloc[:, list(range(1, 21))]
    x_test, y_test = test_data.iloc[:, [0]], test_data.iloc[:, list(range(1, 21))]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # 训练
    temp = x_train.values.tolist()
    X = []
    for i in range(len(temp)):
        X.append(temp[i][0])
    X = tfidf.transform(X)
    y = y_train.values.tolist()
    Y = np.array(y)
    Y = Y.astype(np.int32)
    from scipy import sparse
    X = X.toarray()
    classifier.fit(X, Y)#这样只能单标签，对于多标签分类要采用上述方案 multil_labels_sklearn_xgboost

    temp = x_test.values.tolist()
    X_ts = []
    for i in range(len(temp)):
        X_ts.append(temp[i][0])
    X_Test = tfidf.transform(X_ts)
    Y_Test = y_test.values.tolist()
    Y_Test = np.array(Y_Test)  # 形成一个矩阵
    Y_Test = Y_Test.astype(np.int32)


    predictions = classifier.predict(X_Test)#此时csr_matrix类型
    predictions = predictions.toarray()
    unique, counts = np.unique(predictions, return_counts=True)
    print("preditions=", dict(zip(unique, counts)))
    from sklearn.metrics import accuracy_score
    score = accuracy_score(Y_Test, predictions)
    print(score)


def multil_labels_xgboost_2(x_data, y_data):
    """

    :param x_data:
    :param y_data:
    :return:
    """
    import numpy as np
    import xgboost as xgb

    # 注意需要将其数字化，比如tf-idf或者词向量的方式
    # 这里的tf-idf采用另一种方式，效果是等价的
    # vectorizer = CountVectorizer()
    # tfidftransformer = TfidfTransformer()
    # xgs_tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(x_data))  # 此时返回值是各个词的tf-idf值
    """
    等价写法如下：
    transformer=TfidfVectorizer()
    tfidf_2=transformer.fit_transform(corpus)
    """
    x_train_data, x_test_data, y_train, y_test = train_test_split(x_data, y_data, random_state=42,
                                                                  test_size=0.2, shuffle=True)
    x_train = tfidf.transform(x_train_data)
    x_test = tfidf.transform(x_test_data)

    # y = y_train.tolist()
    # y = np.array(y)
    # Y = Y.astype(np.int32)

    y_train_temp = [0] * len(y_train)# 仅仅是为了能够正常运行，而强制设置的值
    for i in range(len(y_train)):
        if i % 2 == 0:
            y_train_temp[i] = 1
        if i % 3 == 0:
            y_train_temp[i] = 2
    y_test_temp = [0] * len(y_test)
    xg_train = xgb.DMatrix(x_train, label=y_train_temp)#只能单标签分类，因为参数label仅仅支持是1维矩阵或者1维list
    xg_test = xgb.DMatrix(x_test, label=y_test_temp)# 只能支持单标签分类
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 2
    param['num_class'] = 3
    # param['eval_metric'] = 'mlogloss'#'auc',  # 更改为'mlogloss'问题解决
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]## 这步可以不要，用于测试效果

    num_round = 5
    bst = xgb.train(param, xg_train, num_round, watchlist)
    # get prediction
    pred = bst.predict(xg_test)

    print('predicting, classification error=%f' % (
          sum(int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test))))

    # # do the same thing again, but output probabilities
    # param['objective'] = 'multi:softprob'
    # bst = xgb.train(param, xg_train, num_round, watchlist)
    # # Note: this convention has been changed since xgboost-unity
    # # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    # yprob = bst.predict(xg_test).reshape(y_test.shape[0], 6)
    # ylabel = np.argmax(yprob, axis=1)  # return the index of the biggest pro
    #
    # print('predicting, classification error=%f' % (
    #       sum(int(ylabel[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test))))


def multil_labels_sklearn_xgboost_2(x_data, y_data):
    from xgboost import XGBClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import GridSearchCV

    x_train_data, x_test_data, y_train, y_test = train_test_split(x_data, y_data, random_state=42,
                                                                  test_size=0.2, shuffle=True)
    x_train = tfidf.transform(x_train_data)
    x_test = tfidf.transform(x_test_data)
    # one_classifier = XGBClassifier(learning_rate=0.1,
    #                                n_estimators=1000,  # 树的个数--1000棵树建立xgboost
    #                                max_depth=6,  # 树的深度
    #                                min_child_weight=1,  # 叶子节点最小权重
    #                                gamma=0.,  # 惩罚项中叶子结点个数前的参数
    #                                subsample=0.8,  # 随机选择80%样本建立决策树
    #                                colsample_btree=0.8,  # 随机选择80%特征建立决策树
    #                                objective='multi:softmax',  # 指定损失函数
    #                                scale_pos_weight=1,  # 解决样本个数不平衡的问题
    #                                random_state=27  # 随机数
    #                                )
    classifier = OneVsRestClassifier(XGBClassifier(n_jobs=-1, objective='multi:softprob', num_class=20))

    is_singel_model_train = True
    if is_singel_model_train:
        y_train = np.array(y_train)
        y_train = y_train.astype(np.int32)
        y_test = np.array(y_test)# 形成一个矩阵
        y_test = y_test.astype(np.int32)

        classifier.fit(x_train, y_train)
        pred_test = classifier.predict(x_test)
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, pred_test)
        print(score)

    is_grid_search = False
    if is_grid_search:
        vec_x_data = tfidf.transform(x_data)
        vec_y_data = np.array(y_data)
        vec_y_data = vec_y_data.astype(np.int32)
        parameters = {
                        'estimator__scale_pos_weight': [5, 10, 15],# 当正负样本比例为1:10时，scale_pos_weight=10。
                        "estimator__max_depth": [2, 4, 6, 8, 10],#树的深度，默认值为6，典型值3-10
                        "estimator__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],#训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。
                        'estimator__n_estimators': [500, 1000, 2000, 3000],#总共迭代的次数，即决策树的个数
                        'estimator__learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
                     }
        # Best Score: 0.7845903418339664
        # Best Params: {'estimator__colsample_bytree': 0.8, 'estimator__max_depth': 4}

        grid_search_cl = GridSearchCV(classifier, parameters, cv=2, n_jobs=4, verbose=3)
        grid_search_cl.fit(vec_x_data, vec_y_data)
        print("Best Score: ", grid_search_cl.best_score_)
        print("Best Params: ", grid_search_cl.best_params_)


def one_vs_restclassifier():
    """
        数据本身类别偏差，即使全部预测为0，准确率也会很高，所以不能用准确率来评估

    """

    NB_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='elasticnet',
                                                  alpha=1e-4, max_iter=20, random_state=42,
                                                  shuffle=True, n_jobs=-1))),
    ])  # 结果大概0.52

    categories = tiles[1:]
    X_train = train.sentence
    X_test = test.sentence
    prediction_list = []
    test_list = []
    for category in categories:
        print('... Processing {}'.format(category))
        # 用 X_dtm & y训练模型
        NB_pipeline.fit(X_train, train[category])
        # 计算测试准确率
        prediction = NB_pipeline.predict(X_test)
        prediction_int = prediction.astype(np.int32)
        # prediction_array = np.array(prediction, dtype=int32)
        prediction_index = np.where(prediction_int == 1)  # 被分为该类别的样本索引
        if len(prediction_index[0]) > 0:
            print("eixts")
            print(prediction_index[0])
        print(prediction)  # 大部分都是0，只有少数位置会是1
        temp = test[category]
        print(temp.values)
        test_list.append(temp.values)
        prediction_list.append(prediction)
        # 不等于0，且为1的位置
        # prediction_array = np.concatenate([prediction_array, prediction], axis=1)

    prediction_array = np.array(prediction_list)
    print(prediction_array.shape)
    prediction_array = prediction_array.T  # 行方向为样本，列方向为类别
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

    """
    之间对预测结果与真实值进行评估
    """

    score_labor = compute_f1(prediction_array, test_array, tag_dic, tagname_dic)
    print('score_labor', score_labor)


# one_vs_restclassifier()
# binary_relevance(train, test)
# multil_labels_binary_relevance(train, test)
# single_multilabel_model(train, test, train_text, tag_label_array)# 多种单模型的测试

# wordembedding_mlp(alltext_array, tag_label_array)#MLP模型联合预训练词向量的实验，预训练的词向量可能是由于语料过于少，而导致效果反而下降？？？

# multil_labels_xgboost(train, test)# 这个其实不能支持多标签分类
# multil_labels_xgboost_2(train_text, tag_label)# 用其他的xgboost库,一样只能进行单标签分类
multil_labels_sklearn_xgboost(train, test)# 采用onVsRest
# multil_labels_sklearn_xgboost_2(train_text, tag_label)# 采用onVsRest
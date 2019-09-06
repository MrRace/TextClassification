# -*- coding: utf-8 -*-
# @CreatTime    : 2019/8/22 20:00
# @Author  : JasonLiu
# @FileName: main.py
"""
用于线上评测
"""
import os
import pandas as pd
import tensorflow as tf

import bert
from bert import optimization
from bert import tokenization
from bert import modeling
from utils import *


def main():
    # # # 本地服务器测试
    # input_path_labor = "../data/labor/data_small_selected.json"
    # tag_path_labor = "../data/labor//tags.txt"
    # input_path_divorce = "../data/divorce/data_small_selected.json"
    # tag_path_divorce = "../data/divorce/tags.txt"
    # input_path_loan = "../data/loan/data_small_selected.json"
    # tag_path_loan = "../data/loan/tags.txt"
    #
    # output_path_labor = "../output/labor_output.json"
    # output_path_divorce = "../output/divorce_output.json"
    # output_path_loan = "../output/loan_output.json"

    # 线上服务器测试
    input_path_labor = "/input/labor/input.json"
    tag_path_labor = "tags/labor/tags.txt"
    input_path_divorce = "/input/divorce/input.json"
    tag_path_divorce = "tags/divorce/tags.txt"
    input_path_loan = "/input/loan/input.json"
    tag_path_loan = "tags/loan/tags.txt"

    output_path_labor = "/output/labor/output.json"
    output_path_divorce = "/output/divorce/output.json"
    output_path_loan = "/output/loan/output.json"

    BERT_BASE_DIR = "./online_models/bert"  # BERT存放目录
    BERT_VOCAB = os.path.join(BERT_BASE_DIR, 'vocab.txt')
    BERT_CONFIG = os.path.join(BERT_BASE_DIR, 'bert_config.json')

    models_dir = "./online_models"
    num_labels = 20
    learning_rate = 3e-5
    test_batch_size = 4
    max_seq_length = 512
    # 共用的 estimator

    run_config = tf.estimator.RunConfig(
        model_dir=models_dir,
        keep_checkpoint_max=1,  # 只保留一个
        )

    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

    # Compute # train and warmup steps from batch size
    num_train_steps = None
    num_warmup_steps = None
    # 创建模型函数
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        learning_rate=learning_rate,
        init_checkpoint=None,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": test_batch_size})

    tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)

    #*****---------------------预测 labor ------------------------*****
    print('Beginning Labor Predictions!')
    tag_dic, tagname_dic = get_tag_dict(tag_path_labor)
    # 是json格式。这里拿第一阶段的数据进行测试
    fin = open(input_path_labor, "r", encoding="utf-8")
    predict_examples = create_examples(fin, False, "json", tag_dic)  # labels是空的
    test_features = convert_examples_to_features(predict_examples, max_seq_length, tokenizer)

    predict_input_fn = input_fn_builder(features=test_features,
                                        seq_length=max_seq_length,
                                        is_training=False,
                                        drop_remainder=False)

    labor_ckp_path = os.path.join("./online_models/", "labor_best_model.ckpt")
    predictions = estimator.predict(predict_input_fn, checkpoint_path=labor_ckp_path)

    label_columns = []
    for t in tagname_dic:
        label_columns.append(tagname_dic[t])
    output_df, prob_list = create_output(predictions, label_columns)  # DataFrame格式
    # prob_list存放的是每个样本的预测结果，列方向是对每个标签的预测概率
    preds_labels = []
    predic_one_hot_list = []
    for i in range(len(prob_list)):
        row_data = prob_list[i]  # 是一个list
        if len(row_data) != num_labels:
            print("maybe error")
        array = np.array(row_data)
        predic_one_hot = np.where(array > 0.5, 1, 0)
        predic_one_hot_list.append(predic_one_hot)
        indexs = np.where(array >= 0.5)
        temp = []
        if len(indexs[0]) > 0:
            for j in indexs[0]:
                temp.append(j + 1)  # 注意，这里已经+1了
        preds_labels.append(temp)
    generate_pred_file(label_columns, preds_labels, input_path_labor, output_path_labor)
    print('Labor Predictions Success')

    # *****---------------------预测 loan ------------------------*****
    print('Beginning Loan Predictions!')
    tag_dic, tagname_dic = get_tag_dict(tag_path_loan)
    fin = open(input_path_loan, "r", encoding="utf-8")
    predict_examples = create_examples(fin, False, "json", tag_dic)  # labels是空的
    test_features = convert_examples_to_features(predict_examples, max_seq_length, tokenizer)

    predict_input_fn = input_fn_builder(features=test_features,
                                        seq_length=max_seq_length,
                                        is_training=False,
                                        drop_remainder=False)

    loan_ckp_path = os.path.join("./online_models/", "loan_best_model.ckpt")
    predictions = estimator.predict(predict_input_fn, checkpoint_path=loan_ckp_path)

    label_columns = []
    for t in tagname_dic:
        label_columns.append(tagname_dic[t])
    output_df, prob_list = create_output(predictions, label_columns)
    preds_labels = []
    predic_one_hot_list = []
    for i in range(len(prob_list)):
        row_data = prob_list[i]  # 是一个list
        if len(row_data) != num_labels:
            print("maybe error")
        array = np.array(row_data)
        predic_one_hot = np.where(array > 0.5, 1, 0)
        predic_one_hot_list.append(predic_one_hot)
        indexs = np.where(array >= 0.5)
        temp = []
        if len(indexs[0]) > 0:
            for j in indexs[0]:
                temp.append(j + 1)  # 注意，这里已经+1了
        preds_labels.append(temp)
    generate_pred_file(label_columns, preds_labels, input_path_loan, output_path_loan)
    print('Loan Predictions Success')

    # *****---------------------预测 divorce ------------------------*****
    print('Beginning Divorce Predictions!')
    tag_dic, tagname_dic = get_tag_dict(tag_path_divorce)
    fin = open(input_path_divorce, "r", encoding="utf-8")
    predict_examples = create_examples(fin, False, "json", tag_dic)  # labels是空的
    test_features = convert_examples_to_features(predict_examples, max_seq_length, tokenizer)

    predict_input_fn = input_fn_builder(features=test_features,
                                        seq_length=max_seq_length,
                                        is_training=False,
                                        drop_remainder=False)

    divorce_ckp_path = os.path.join("./online_models/", "divorce_best_model.ckpt")
    predictions = estimator.predict(predict_input_fn, checkpoint_path=divorce_ckp_path)

    label_columns = []
    for t in tagname_dic:
        label_columns.append(tagname_dic[t])
    output_df, prob_list = create_output(predictions, label_columns)
    preds_labels = []
    predic_one_hot_list = []
    for i in range(len(prob_list)):
        row_data = prob_list[i]  # 是一个list
        if len(row_data) != num_labels:
            print("maybe error")
        array = np.array(row_data)
        predic_one_hot = np.where(array > 0.5, 1, 0)
        predic_one_hot_list.append(predic_one_hot)
        indexs = np.where(array >= 0.5)
        temp = []
        if len(indexs[0]) > 0:
            for j in indexs[0]:
                temp.append(j + 1)  # 注意，这里已经+1了
        preds_labels.append(temp)
    generate_pred_file(label_columns, preds_labels, input_path_divorce, output_path_divorce)
    print('Divorce Predictions Success')


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# @CreatTime    : 2019/8/14 9:36
# @Author  : JasonLiu
# @FileName: multi-label-classification-bert.py

import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
import configuration as cfig
import eval
##install bert if not already done
##!pip install bert-tensorflow

import bert
from bert import optimization
from bert import tokenization
from bert import modeling
import pdb
from utils import *


def main():

    if cfig.do_process:
        raw_filename = os.path.join(cfig.raw_data_base_dir, cfig.raw_data_filename)
        tag_dic, tagname_dic = get_tag_dict(cfig.tag_file)
        LABEL_COLUMNS = []
        for t in tagname_dic:
            LABEL_COLUMNS.append(tagname_dic[t])
        print("labels=", LABEL_COLUMNS)
        titles = ["sentence"]
        titles.extend(LABEL_COLUMNS)
        process_data(raw_filename, tag_dic, titles)
        exit()
    # 共用的 estimator
    run_config = tf.estimator.RunConfig(
        model_dir=cfig.models_dir,
        save_summary_steps=cfig.save_summary_steps,
        keep_checkpoint_max=1,  # 只保留一个
        save_checkpoints_steps=cfig.save_checkpoints_steps)

    bert_config = modeling.BertConfig.from_json_file(cfig.BERT_CONFIG)

    # Compute # train and warmup steps from batch size
    num_train_steps = None
    num_warmup_steps = None
    is_train_data_fixed = True

    if cfig.do_train:
        ##change path accordingly
        train_data_path = os.path.join(cfig.data_base_dir, cfig.train_file_name)
        train = pd.read_csv(train_data_path, delimiter="\t")
        # print(train.head())
        if cfig.is_use_all_train_data:
            x_train = train  # 采用全部的训练数据集进行训练,此时则没有评估阶段，或者说评估的结果是不可信的
        else:
            if is_train_data_fixed:
                # 训练集固定的
                if not cfig.is_use_all_train_data:
                    x_train, x_val = train_test_split(train, random_state=42, train_size=cfig.train_val_ratio, shuffle=True)
                    # 查看是否每次都是一样的,x_train[2], x_val[2]是否每次都一样。应该是相同的，random_state值固定则每次都得到同样的划分
                    # 为了方便和PyTorch版本对比，将x_train, x_val进行保存。所以，PyTorch版本只需要直接使用train_spilt_0.9.csv
                    # 和dev_spilt_0.9.csv即可
                    train_path = os.path.join(cfig.data_base_dir, "train_spilt_0.9.csv")
                    dev_path = os.path.join(cfig.data_base_dir, "dev_spilt_0.9.csv")
                    x_train.to_csv(train_path, sep="\t", encoding="utf-8", index=0)  # 不保留行索引，因为已经有了
                    x_val.to_csv(dev_path, sep="\t", encoding="utf-8", index=0)
                else:
                    # 没有使用dev data
                    pass
            else:
                # 训练集和dev set 每次都不同
                x_train, x_val = train_test_split(train, train_size=cfig.train_val_ratio, shuffle=True)
        train_examples = create_examples(x_train)
        num_train_steps = int(len(train_examples) / cfig.train_batch_size * cfig.num_train_epochs)
        num_warmup_steps = int(num_train_steps * cfig.warmup_proportion)
    # 创建模型函数
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=cfig.num_labels,
        init_checkpoint=cfig.BERT_INIT_CHKPNT,
        learning_rate=cfig.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": cfig.train_batch_size})

    tokenization.validate_case_matches_checkpoint(True, cfig.BERT_INIT_CHKPNT)
    tokenizer = tokenization.FullTokenizer(vocab_file=cfig.BERT_VOCAB, do_lower_case=True)
    # print(tokenizer.tokenize("这是一个例子而已，请注意"))
    if not os.path.exists(cfig.output_dir):
        os.makedirs(cfig.output_dir)
    if cfig.do_train:
        # 进行训练

        train_file = os.path.join(cfig.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            open(train_file, 'w').close()

        # 训练集转为features
        file_based_convert_examples_to_features(train_examples, cfig.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", cfig.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        # 创建导入函数
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=cfig.max_seq_length,
            is_training=True,
            drop_remainder=True)

        print('Beginning Training!')
        current_time = datetime.now()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("Training took time ", datetime.now() - current_time)

        # val_data_path = os.path.join(cfig.data_base_dir, 'dev.csv')
        # x_val = pd.read_csv(val_data_path, delimiter="\t")
        if not cfig.is_use_all_train_data:
            print("Beginning do evaluation!")
            # 如果使用全部的train进行训练，此时是没有划分出dev data进行评估之用的
            eval_file = os.path.join(cfig.output_dir, "eval.tf_record")
            if not os.path.exists(eval_file):
                open(eval_file, 'w').close()

            eval_examples = create_examples(x_val)
            file_based_convert_examples_to_features(eval_examples, cfig.max_seq_length, tokenizer, eval_file)

            # This tells the estimator to run through the entire set.
            eval_steps = None

            eval_drop_remainder = False
            eval_input_fn = file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=cfig.max_seq_length,
                is_training=False,
                drop_remainder=False)

            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
            output_eval_file = os.path.join(cfig.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

    if cfig.do_predict:
        print('Beginning Predictions!')
        tag_dic, tagname_dic = get_tag_dict(cfig.tag_file)
        # 进行prediction
        # 事先从原始数据中划分为test data,使其保持独立性
        test_data_path = os.path.join(cfig.data_base_dir, cfig.test_file_name)  # 这里dev.csv

        if cfig.test_data_format == "csv":
            test = pd.read_csv(test_data_path, delimiter="\t")
            # x_test = test[:10000]  # testing a small sample
            x_test = test
            x_test = x_test.reset_index(drop=True)
            predict_examples = create_examples(x_test, True)  # 注意，此时并不是从原始的json格式中读取数据
        else:
            # 是json格式。这里拿第一阶段的数据进行测试
            fin = open(test_data_path, "r", encoding="utf-8")
            predict_examples = create_examples(fin, True, cfig.test_data_format, tag_dic)

        test_features = convert_examples_to_features(predict_examples, cfig.max_seq_length, tokenizer)

        current_time = datetime.now()
        predict_input_fn = input_fn_builder(features=test_features,
                                            seq_length=cfig.max_seq_length,
                                            is_training=False,
                                            drop_remainder=False)
        #
        ckp_path = os.path.join("./model_aug_bacthsize_4_learningrate_3e-5_epoch_3_laborscore_73_44", "best_model.ckpt")
        predictions = estimator.predict(predict_input_fn)  # , checkpoint_path=ckp_path
        print("Prediction took time ", datetime.now() - current_time)

        label_columns = []
        for t in tagname_dic:
            label_columns.append(tagname_dic[t])
        output_df, prob_list = create_output(predictions, label_columns)  # DataFrame格式
        # prob_list存放的是每个样本的预测结果，列方向是对每个标签的预测概率
        preds_labels = []
        predic_one_hot_list = []
        for i in range(len(prob_list)):
            row_data = prob_list[i]  # 是一个list
            if len(row_data) != cfig.num_labels:
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
        # 与标准答案进行对比
        true_tags_count = 0
        predic_tags_count = 0
        all_qual_num = 0
        test_list = []
        for i in range(len(predict_examples)):
            # 遍历数据集中的标准
            one_exam = predict_examples[i]
            one_tags = one_exam.labels  # 真实labels,是一个arrya，值为0和1
            test_list.append(one_tags)
            # 和predic_one_hot_list逐行对比
            pred_rs = np.array(predic_one_hot_list[i])
            # pdb.set_trace()
            # predic_labels = preds_labels[i]
            if 1 in one_tags:
                # 存在1的时候，说明真实标签非空，统计有标签结果的样本数量
                true_tags_count = true_tags_count + 1
            if 1 in pred_rs:
                predic_tags_count = predic_tags_count + 1
            # 存在
            if 1 in one_tags and 1 in pred_rs and (one_tags == pred_rs).all():
                all_qual_num = all_qual_num + 1
        print("true_count={},predict_count={}".format(true_tags_count, predic_tags_count))
        print("all_qual_num=", all_qual_num)
        # out_filename = "{}_output.json".format(cfig.task_type_name)
        # outf_file = os.path.join(cfig.output_dir, out_filename)
        # inf_path = os.path.join(labor_data_path, data_filename)
        # generate_pred_file(label_columns, labor_preds, inf_path, outf_file)

        # 从2个矩阵的角度计算score。默认计算得分的方式，是将结果填入填入原始json格式中的label字段，再与标准的比对
        prediction_array = np.array(predic_one_hot_list)
        print(prediction_array.shape)
        # prediction_array = prediction_array.T  # 行方向为样本，列方向为类别
        # print(prediction_array.shape)

        test_array = np.array(test_list)
        print(test_array.shape)
        # test_array = test_array.T
        score_labor = eval.compute_f1(prediction_array, test_array, tag_dic, tagname_dic)
        print('score_labor', score_labor)

        # merged_df = pd.concat([x_test, output_df], axis=1)
        # submission = merged_df.drop(['sentence'], axis=1)  # 去掉文本内容列
        # predict_file = os.path.join(cfig.output_dir, "predict_result.csv")
        # submission.to_csv(predict_file, index=False)
        # print(submission.head())


if __name__ == "__main__":
    main()

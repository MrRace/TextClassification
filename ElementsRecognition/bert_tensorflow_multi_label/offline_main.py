# -*- coding: utf-8 -*-
# @CreatTime    : 2019/8/17 16:51
# @Author  : JasonLiu
# @FileName: offline_main.py

"""
用于离线评测
"""
import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
import configuration as cfig
import eval

import bert
from bert import optimization
from bert import tokenization
from bert import modeling
import pdb
from utils import *


def main():
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

    if cfig.do_predict:
        print('Beginning Labor Predictions!')
        tag_dic, tagname_dic = get_tag_dict(cfig.tag_file)
        test_data_path = os.path.join(cfig.data_base_dir, cfig.test_file_name)

        # 是json格式。这里拿第一阶段的数据进行测试
        fin = open(test_data_path, "r", encoding="utf-8")
        predict_examples = create_examples(fin, False, "json", tag_dic)  # labels是空的
        test_features = convert_examples_to_features(predict_examples, cfig.max_seq_length, tokenizer)

        predict_input_fn = input_fn_builder(features=test_features,
                                            seq_length=cfig.max_seq_length,
                                            is_training=False,
                                            drop_remainder=False)

        ckp_path = os.path.join("./model_labor/", "best_model.ckpt")
        predictions = estimator.predict(predict_input_fn, checkpoint_path=ckp_path)

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
        out_filename = "{}_output.json".format(cfig.task_type_name)
        outf_file = os.path.join(cfig.output_dir, out_filename)
        inf_path = test_data_path
        generate_pred_file(label_columns, preds_labels, inf_path, outf_file)


if __name__ == "__main__":
    main()

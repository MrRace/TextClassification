# -*- coding: utf-8 -*-
# @CreatTime    : 2019/9/20 11:46
# @Author  : JasonLiu
# @FileName: test_tfserving.py
import requests
import json
import tensorflow as tf
import collections
import pdb
import numpy as np
from bert import tokenization
from utils import create_examples_text_list, convert_single_example


def test_request():
    label_ids = 20*[0]
    input_ids = 512*[1]
    input_mask = 512*[1]
    segment_ids = 512*[1]
    data_dict_temp = {
            'label_ids': label_ids,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
    }
    data_list = []
    data_list.append(data_dict_temp)

    data = json.dumps({"signature_name": "serving_default", "instances": data_list})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/cail_elem:predict', data=data, headers=headers)
    print(json_response.text)
    predictions = json.loads(json_response.text)['predictions']
    print(predictions)


def request_from_raw_text():
    """

    :return:
    """
    BERT_VOCAB = "/home/data1/ftpdata/pretrain_models/bert_tensoflow_version/bert-base-chinese-vocab.txt"
    text_list = ["权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。",
                 "权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 "2012年11月30日，原债权人工行锦州市分行向保证人锦州锅炉有限责任公司发出督促履行保证责任通知书，要求其履行保证责任，"
                 "2004年11月18日，原债权人工行锦州市分行采用国内挂号信函的方式向保证人锦州锅炉有限责任公司邮寄送达中国工商银行辽宁省分行督促履行保证责任通知书，"  # LN4
                 "锦州市凌河区公证处相关公证人员对此过程进行了公证。"
                 ]
    data_list = []
    tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)
    predict_examples = create_examples_text_list(text_list)
    # pdb.set_trace()
    for (ex_index, example) in enumerate(predict_examples):
        feature = convert_single_example(ex_index, example,
                                         512, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = {}
        features["input_ids"] = feature.input_ids
        features["input_mask"] = feature.input_mask
        # pdb.set_trace()
        features["segment_ids"] = feature.segment_ids
        if isinstance(feature.label_ids, list):
            label_ids = feature.label_ids
        else:
            label_ids = feature.label_ids[0]
        features["label_ids"] = label_ids
        # tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        data_list.append(features)

    # pdb.set_trace()
    data = json.dumps({"signature_name": "serving_default", "instances": data_list})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/cail_elem:predict', data=data, headers=headers)
    # print(json_response.text)
    # pdb.set_trace()
    predictions = json.loads(json_response.text)['predictions']
    # print(predictions)
    for p in range(len(predictions)):
        p_list = predictions[p]
        label_index = np.argmax(p_list)
        print("content={},label={}".format(text_list[p], label_index+1))
    print("total number=", len(text_list))

request_from_raw_text()

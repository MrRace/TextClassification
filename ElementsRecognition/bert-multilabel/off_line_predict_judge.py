# -*- coding: utf-8 -*-
# @CreatTime    : 2019/7/28 17:55
# @Author  : JasonLiu
# @FileName: off_line_predict_judge.py
"""
离线测试运行及其结果评估
"""
import json
import os
import argparse
from judger import Judger
import torch
import numpy as np
from sklearn.externals import joblib


from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, \
    BertForMaskedLM, BertForSequenceClassification
from pathlib import Path
import torch
import re
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import collections
import os
import pdb
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import pdb


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_label_id(my_dic, q):
    return my_dic[q]


def get_tag_dict(tags_path):
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


model_state_dict = None

args = {
    "train_size": -1,
    "val_size": -1,
    "no_cuda": False,
    "max_seq_length": 512,#可以修改，默认512
    "do_eval": True,
    "do_lower_case": False,
    "eval_batch_size": 128,
    "num_train_epochs": 4.0,
    "warmup_proportion": 0.1,
    "num_gpus": 1,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=20):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None, my_labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
        self.my_labels = my_labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class MultiLabelTextProcessor(DataProcessor):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None

    def get_test_examples(self, data_dir, data_file_name, data_format="json", labor_tagname_dic=""):
        """
        (1)线上的数据是以json格式存储，所以需要支持从json文件中读取
        (2)还需要支持从csv中读取数据，因为在离线时候还切分出一个test data
        :param data_dir:
        :param data_file_name:
        :param size:
        :return:
        """
        # data_df = pd.read_csv(os.path.join(data_dir, data_file_name), delimiter="\t")
        if data_format == "json":
            data_fin = open(os.path.join(data_dir, data_file_name), 'r', encoding='utf8')
        else:
            data_fin = pd.read_csv(os.path.join(data_dir, data_file_name), delimiter="\t")
        return self._create_examples(data_fin, data_format, "test", True)  # 这里的test data是带有标签的

    def get_labels(self):
        """See base class."""
        if self.labels == None:
            pass
            # self.labels = list(pd.read_csv(os.path.join(self.data_dir, "classes.txt"), header=None)[0].values)
        return self.labels

    def _create_examples(self, fin, format, set_type, labels_available=True):
        """Creates examples for the training and dev sets."""
        if format == "json":
            line_data = fin.readline()
            line_num = 1
            examples = []
            while line_data:
                d = json.loads(line_data)
                for sent in d:
                    sentence = sent['sentence']
                    line_num = line_num + 1
                    guid = line_num
                    text_a = sentence
                    temp_labels = sent['labels']# LB1 - LB20
                    if not temp_labels:
                        temp_labels = []
                    examples.append(InputExample(guid=guid, text_a=text_a, my_labels=temp_labels))
                line_data = fin.readline()
            self.example_nums = line_num
        else:
            # 读取的是csv
            examples = []
            for (i, row) in enumerate(fin.values):
                guid = row[0]
                text_a = row[1]
                temp_labels = []
                if labels_available:
                    labels = row[2:]
                    # 如果没有1，全部是0，直接给空[]??
                    if (labels == 1).any():
                        # 存在一个1
                        # labels = labels.tolist()
                        # 此时的labels是0/1的list，需要转为真实的形如LB1 LB2这样的
                        # 找出1的index
                        indexs = np.where(labels == 1)
                        for j in indexs[0]:
                            temp_labels.append(labor_tagname_dic[j])
                        # pdb.set_trace()
                    else:
                        pass
                else:
                    pass
                examples.append(InputExample(guid=guid, text_a=text_a, my_labels=temp_labels))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        labels_ids = []
        if example.labels:
            for label in example.labels:
                labels_ids.append(float(label))

        #         label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=labels_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class Predictor(object):
    def __init__(self, model, device):
        self.model = model
        self.model.to(device)
        self.batch_size = 1

    def predict_sentence(self, test_examples, label_list, tokenizer, device):
        """
        每个句子？还是一次处理一个batch???
        :param vec:
        :param device:
        :return:
        """
        print("***** Running prediction *****")
        print("Num examples = ", len(test_examples))
        print("Batch size = ", args['eval_batch_size'])

        test_features = convert_examples_to_features(test_examples, label_list, args['max_seq_length'], tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])

        all_logits = None
        model.eval()
        nb_eval_steps, nb_eval_examples = 0, 0
        for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
            input_ids, input_mask, segment_ids = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        # 注意查看预测结果，查看几条
        # 根据all_logits返回对应的labels
        # pdb.set_trace()
        all_tags = []
        [rows, _] = all_logits.shape
        for i in range(rows):
            row_data = all_logits[i, :]
            indexs = np.where(row_data >= 0.5)
            temp = []
            if len(indexs[0]) > 0:
                for j in indexs[0]:
                    temp.append(j + 1)#注意，这里已经+1了
            all_tags.append(temp)
        return all_tags
        # return pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns=label_list), left_index=True,
        #                 right_index=True)

    def predict(self, content):
        # vec = self.tfidf.transform([fact])
        # ans = self.predict_tag(vec)#预测结果
        # print(ans)
        ans = ""
        return ans


def generate_pred_file(tags_list, pred_list, inf_path, outf_path):
    with open(inf_path, 'r', encoding='utf-8') as inf, open(
            outf_path, 'w', encoding='utf-8') as outf:
        line_num = 0
        for line in inf.readlines():
            pre_doc = json.loads(line)
            predict_doc = []
            for ind in range(len(pre_doc)):
                pred_sent = pre_doc[ind]
                pre_content = pre_doc[ind]['sentence']
                # prd此时是一个矩阵
                pred_label = pred_list[line_num]#这里的label是纯数字
                line_num = line_num + 1
                label_names = []
                for label in pred_label:
                    label_names.append(tags_list[label - 1])
                pred_sent['labels'] = label_names
                predict_doc.append(pred_sent)
            json.dump(predict_doc, outf, ensure_ascii=False)
            outf.write('\n')


def process_by_task(law_task_name, stage_num):
    """"""
    pass


if __name__ == '__main__':
    stage_num = 4#以第一阶段的数据进行评估
    # 生成labor领域的预测文件
    # loan ， divorce ， labor
    parser = argparse.ArgumentParser()
    ## Required parameters
    result_file = open("merge_data_aug_result.txt", 'a', encoding='utf-8')
    parser.add_argument("--task_type_name", default=None, type=str, required=True,
                        help="input the law data name, labor | loan | divorce")
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="The pretrained model, bert | wwm | wwm_ext")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--model_postfix", default="", type=str,
                        help="The postfix name of the model")
    parser.add_argument("--data_prefix", default="", type=str,
                        help="The prefix name of the data")

    init_args = parser.parse_args()

    task_type_name = init_args.task_type_name  # "labor"
    model_name = init_args.model_name  # "bert"  # wwm_ext  wwm  bert
    model_postfix = init_args.model_postfix  # 一般是numepoch_3的后缀
    learning_rate = init_args.learning_rate
    data_prefix = init_args.data_prefix
    data_format = "json"  # json(线上数据) 或者 csv(test数据集)
    print('predict {}...'.format(task_type_name))
    labor_tags_list = []
    # 数据格式可能是csv
    if stage_num == 1:
        labor_data_dir = "../data/"
        data_filename = 'data_small_selected.json'
    elif stage_num == 3:
        # test.csv格式
        # 和TensorFlow使用相同的数据集，以方便比较
        labor_data_dir = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/bert_tensorflow_multi_label/data/"
        data_filename = "test.csv"
        data_format = "csv"
    elif stage_num == 4:
        # test.json格式
        if data_prefix:
            labor_data_dir = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/data/CAIL2019-FE-big"
            # data_filename = "{}_test.json".format(data_prefix)
            data_filename = "merge_processed_test.json"
        else:
            labor_data_dir = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/bert_tensorflow_multi_label/data/"
            data_filename = "test.json"
        # 格式是json
    else:
        labor_data_dir = "../data/CAIL2019-FE-big/"
        data_filename = "train_selected.json"

    labor_data_path = os.path.join(labor_data_dir, task_type_name)
    if stage_num == 3 or stage_num == 4:
        # 因为该位置没有tag文件
        labor_tag_file = os.path.join("../data/CAIL2019-FE-big/" + task_type_name, "tags.txt")
    else:
        labor_tag_file = os.path.join(labor_data_path, "tags.txt")

    with open(labor_tag_file, 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            labor_tags_list.append(line.strip())
    labor_tag_dic, labor_tagname_dic = get_tag_dict(labor_tag_file)

    labor_model_dir = "./models_mergedata_aug"#存放微调好的模型、bert_json和vocab。这3个文件！   models_sharedatat_tf
    if data_prefix:
        model_filename = "{}_{}_{}_learnrate_{}{}.bin".format(task_type_name, data_prefix, model_name, learning_rate,
                                                              model_postfix)
    else:
        model_filename = "{}_{}_learnrate_{}{}.bin".format(task_type_name, model_name, learning_rate, model_postfix)  # 0.6538
    result_file.write("model_filename={}\n".format(model_filename))
    model_file = os.path.join(labor_model_dir, model_filename)
    model_state_dict = torch.load(model_file)
    model = BertForMultiLabelSequenceClassification.from_pretrained(labor_model_dir, num_labels=20,
                                                                    state_dict=model_state_dict)
    device = torch.device("cuda:2")
    prd = Predictor(model, device)
    # 读取测试数据
    # labor_inf_path = os.path.join(labor_data_path, test_data_filename)
    predict_processor = MultiLabelTextProcessor(labor_data_path)
    test_examples = predict_processor.get_test_examples(labor_data_path, data_filename, data_format, labor_tagname_dic)
    """查看多少个有tag
    """
    # Hold input data for returning it
    print("test_examples len=", len(test_examples))#已经将label也加入其中
    # pdb.set_trace()
    input_data = [{'id': input_example.guid, 'comment_text': input_example.text_a} for input_example in test_examples]
    tokenizer = BertTokenizer.from_pretrained(labor_model_dir, do_lower_case=args['do_lower_case'])#其实这个token是可以复用
    labor_preds = prd.predict_sentence(test_examples, labor_tags_list, tokenizer, device)
    # print("shape of labor_preds=", labor_preds.shape)
    true_tags_count = 0
    predic_tags_count = 0
    all_qual_num = 0
    for i in range(len(test_examples)):
        one_exam = test_examples[i]
        one_text = one_exam.text_a
        one_tags = one_exam.my_labels
        predic_labels = labor_preds[i]
        if one_tags:
            true_tags_count = true_tags_count + 1
        if predic_labels:
            predic_tags_count = predic_tags_count + 1
        if one_tags or predic_labels:
            predic_labels_names = []
            if predic_labels:
                for j in predic_labels:
                    predic_labels_names.append(labor_tags_list[j - 1])
            # print("true={},predict={}".format(one_tags, predic_labels_names))
            # pdb.set_trace()
            if set(one_tags) == set(predic_labels_names):
                all_qual_num = all_qual_num + 1
            # pdb.set_trace()
    result_file.write("true_count={},predict_count={},all_qual_num={}\n".format(true_tags_count,
                                                                              predic_tags_count, all_qual_num))
    # pdb.set_trace()
    outf_path = '../output/'
    out_filename = "{}_output.json".format(task_type_name)
    outf_file = os.path.join(outf_path, out_filename)
    inf_path = os.path.join(labor_data_path, data_filename)
    generate_pred_file(labor_tags_list, labor_preds, inf_path, outf_file)

    # 对结果进行评估
    judger_labor = Judger(tag_path=labor_tag_file)
    reslt_labor = judger_labor.test(truth_path=inf_path,
                                    output_path=outf_file)
    score_labor = judger_labor.gen_score(reslt_labor)
    result_file.write('score_{}={}\n\n'.format(model_filename, score_labor))

    exit()

    # 生成divorce领域的预测文件
    print('predict_divorce...')
    tags_list = []
    with open('../../data/divorce/tags.txt', 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
    prd = Predictor('model_divorce/')
    inf_path = '../../data/divorce/data_small_selected.json'
    outf_path = '../../output/divorce_output.json'
    generate_pred_file(tags_list, prd, inf_path, outf_path)

    # 生成loan领域的预测文件
    print('predict_loan...')
    tags_list = []
    with open('../../data/loan/tags.txt', 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
    prd = Predictor('model_loan/')
    inf_path = '../../data/loan/data_small_selected.json'
    outf_path = '../../output/loan_output.json'
    generate_pred_file(tags_list, prd, inf_path, outf_path)

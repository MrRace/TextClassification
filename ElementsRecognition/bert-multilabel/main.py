# -*- coding: utf-8 -*-
# @CreatTime    : 2019/7/28 17:53
# @Author  : JasonLiu
# @FileName: online_main.py

import json
import sys
import torch
import os
import numpy as np
import jieba
from tqdm import tqdm, trange
from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, \
    BertForMaskedLM, BertForSequenceClassification
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.externals import joblib



module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_state_dict = None

args = {
    "train_size": -1,
    "val_size": -1,
    "no_cuda": False,
    "max_seq_length": 512,
    "do_eval": True,
    "do_lower_case": False,
    "eval_batch_size": 64,
    "learning_rate": 3e-5,
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
    def __init__(self, guid, text_a, text_b=None, my_labels=None, labels=None):
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

    def get_test_examples(self, data_file_name, size=-1):
        data_fin = open(data_file_name, 'r', encoding='utf-8')
        return self._create_examples(data_fin, "test", False)

    def _create_examples(self, fin, set_type, labels_available=True):
        """Creates examples for the training and dev sets."""
        line_data = fin.readline()
        line_num = 1
        examples = []
        while line_data:
            d = json.loads(line_data)
            for sent in d:
                sentence = sent['sentence']#线上是Unicode编码
                sentence = sentence.encode('utf-8').decode('unicode_escape')
                logger.info("sentence={}".format(sentence))
                line_num = line_num + 1
                guid = line_num
                text_a = sentence
                temp_labels = sent['labels']# LB1 - LB20
                if not temp_labels:
                    temp_labels = []
                examples.append(InputExample(guid=guid, text_a=text_a, my_labels=temp_labels))
            line_data = fin.readline()
        self.example_nums = line_num
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    # logger.info("examples len={}".format(len(test_examples)))
    features = []
    line_num_count = 1
    for (ex_index, example) in enumerate(examples):
        logger.info("line_num_count={}".format(line_num_count))
        line_num_count = line_num_count + 1
        logger.info("text_a={}".format(example.text_a))
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

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
    def __init__(self):
        self.batch_size = 1

    def predict_sentence(self, test_features, device):
        logger.info("***** Running prediction *****")
        logger.info("  Batch size = %d", args['eval_batch_size'])

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
        all_tags = []
        [rows, _] = all_logits.shape
        for i in range(rows):
            row_data = all_logits[i, :]
            indexs = np.where(row_data >= 0.3)
            temp = []
            if len(indexs[0]) > 0:
                for j in indexs[0]:
                    temp.append(j + 1)#注意，这里已经+1了
            all_tags.append(temp)
        return all_tags


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


class Predictor_old(object):
    def __init__(self, model_dir):
        self.tfidf = joblib.load(model_dir + 'tfidf.model')
        self.tag = joblib.load(model_dir + 'tag.model')
        self.batch_size = 1
        self.cut = jieba

    def predict_tag(self, vec):
        y = self.tag.predict(vec)
        #y = y.toarray().astype(np.int32)#.tolist()
        indexs = np.where(y == 1)
        if len(indexs[0]) > 0:
            temp = []
            for i in indexs[1]:
                temp.append(i + 1)
            return temp
        else:
            return []

    def predict(self, content):
        fact = ' '.join(self.cut.cut(str(content)))
        vec = self.tfidf.transform([fact])
        ans = self.predict_tag(vec)
        return ans


def generate_pred_file_old(tags_list, prd, inf_path, outf_path):
    with open(inf_path, 'r', encoding='utf-8') as inf, open(
            outf_path, 'w', encoding='utf-8') as outf:
        for line in inf.readlines():
            pre_doc = json.loads(line)
            predict_doc = []
            for ind in range(len(pre_doc)):
                pred_sent = pre_doc[ind]
                pre_content = pre_doc[ind]['sentence']
                pred_label = prd.predict(pre_content)#
                label_names = []
                for label in pred_label:
                    label_names.append(tags_list[label - 1])
                pred_sent['labels'] = label_names
                predict_doc.append(pred_sent)
            json.dump(predict_doc, outf, ensure_ascii=False)
            outf.write('\n')


if __name__ == '__main__':
    # # 本地服务器测试
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

    # 生成labor领域的预测文件
    logger.info("***** labor prediction *****")
    labor_tags_list = []
    with open(tag_path_labor, 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            labor_tags_list.append(line.strip())

    model_dir = "./models"#存放微调好的模型、bert_json和vocab。这3个文件！
    labor_data_path = "/input/labor/"
    predict_processor = MultiLabelTextProcessor(labor_data_path)
    test_examples = predict_processor.get_test_examples(input_path_labor, size=-1)
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=args['do_lower_case'])
    labor_test_features = convert_examples_to_features(test_examples, labor_tags_list, args['max_seq_length'], tokenizer)#报错？？？
    logger.info("labor Num examples = %d", len(test_examples))
    # 加载训练好的模型进行预测
    model_filename = "labor_finetuned_pytorch_model.bin"
    model_file = os.path.join(model_dir, model_filename)
    model_state_dict = torch.load(model_file)
    model = BertForMultiLabelSequenceClassification.from_pretrained(model_dir, num_labels=20,
                                                                    state_dict=model_state_dict)
    device = torch.device("cuda:0")
    model.to(device)
    prd = Predictor()
    labor_preds = prd.predict_sentence(labor_test_features, device)
    generate_pred_file(labor_tags_list, labor_preds, input_path_labor, output_path_labor)

    # 生成divorce领域的预测文件
    logger.info("***** divorce prediction *****")
    divorce_tags_list = []
    with open(tag_path_divorce, 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            divorce_tags_list.append(line.strip())

    test_examples = predict_processor.get_test_examples(input_path_divorce, size=-1)
    divorce_test_features = convert_examples_to_features(test_examples, divorce_tags_list, args['max_seq_length'], tokenizer)  # 报错？？？
    logger.info("divorce Num examples = %d", len(test_examples))
    model_filename = "divorce_finetuned_pytorch_model.bin"
    divorce_model_file = os.path.join(model_dir, model_filename)
    model_state_dict = torch.load(divorce_model_file)
    model = BertForMultiLabelSequenceClassification.from_pretrained(model_dir, num_labels=20,
                                                                    state_dict=model_state_dict)
    model.to(device)
    prd = Predictor()
    divorce_data_path = "/input/divorce/"
    predict_processor = MultiLabelTextProcessor(divorce_data_path)
    divorce_preds = prd.predict_sentence(divorce_test_features, device)
    generate_pred_file(divorce_tags_list, divorce_preds, input_path_divorce, output_path_divorce)

    # 生成loan领域的预测文件
    logger.info("***** loan prediction *****")
    tags_list = []
    with open(tag_path_loan, 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
    prd = Predictor_old('./models/model_loan/')
    generate_pred_file_old(tags_list, prd, input_path_loan, output_path_loan)

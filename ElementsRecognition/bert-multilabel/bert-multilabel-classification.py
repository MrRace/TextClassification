# -*- coding: utf-8 -*-
# @CreatTime    : 2019/7/27 16:21
# @Author  : JasonLiu

from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, \
    BertForMaskedLM, BertForSequenceClassification
from pathlib import Path
import torch
import re
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
# from fastai.text import Tokenizer, Vocab# fastai1.0版本以上需要python3.6,好像整个fastai都是如此，即使是低版本也是
import pandas as pd
import collections
import os
import argparse
from tqdm import tqdm, trange
import sys
import random
import numpy as np
# import apex
import pdb
from sklearn.model_selection import train_test_split

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from sklearn.metrics import roc_curve, auc

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam

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


parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--task_type_name", default=None, type=str, required=True,
                    help="input the law data name, labor | loan | divorce")
parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
parser.add_argument("--model_name", default=None, type=str, required=True,
                    help="The pretrained model, bert | wwm | wwm_ext")
parser.add_argument("--data_prefix", default="", type=str,
                    help="The prefix name of the data")

init_args = parser.parse_args()

task_type_name = init_args.task_type_name  #"labor"  # 需要对3种数据进行训练： loan ， divorce ， labor
data_prefix = init_args.data_prefix  # 当使用到merge数据集的时候需要
stage_num = 3
is_use_aug_data = False  # 是否使用增强的数据集
is_do_model_eval = True  # 是否使用dev data选出最好的模型。如果使用merge_processed_train.csv，则不需要dev data
if data_prefix:
    # 如果指定为merge data，则不需dev data
    is_do_model_eval = False

if stage_num == 1:
    data_dir_path = "../data/" + task_type_name
elif stage_num == 3:
    # 采用与TensorFlow版本相同的数据进行对比
    if data_prefix:
        data_dir_path = os.path.join("/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/data/CAIL2019-FE-big",
                                     task_type_name)
    else:
        data_dir_path = os.path.join("/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/"
                                     "bert_tensorflow_multi_label/data", task_type_name)
else:
    data_dir_path = "../data/CAIL2019-FE-big/" + task_type_name

if stage_num == 3:
    # 还是使用
    tag_file = os.path.join("../data/CAIL2019-FE-big/" + task_type_name, "tags.txt")
else:
    tag_file = os.path.join(data_dir_path, "tags.txt")
tag_dic, tagname_dic = get_tag_dict(tag_file)

tiles = []
for t in tagname_dic:
    tiles.append(tagname_dic[t])

DATA_PATH = Path(data_dir_path)
DATA_PATH.mkdir(exist_ok=True)

# 临时数据存放在该
PATH = Path("./output")
PATH.mkdir(exist_ok=True)

model_state_dict = None

# BERT_PRETRAINED_PATH = Path('../trained_model/')
# BERT_PRETRAINED_PATH = Path('/home/data1/ftpdata/pretrain_models/bert_pytorch_version/bert-base-chinese/')#默认的bert预训练模型
model_name = init_args.model_name  # bert  wwm  wwm_ext
if model_name == "wwm_ext":
    BERT_PRETRAINED_PATH = Path('/home/data1/ftpdata/pretrain_models/chinese_bert_wwm/chinese_wwm_ext_pytorch/')#WWM-ext预训练模型
elif model_name == "wwm":
    BERT_PRETRAINED_PATH = Path('/home/data1/ftpdata/pretrain_models/chinese_bert_wwm/chinese_wwm_pytorch/')# WWM预训练模型
else:
    BERT_PRETRAINED_PATH = Path('/home/data1/ftpdata/pretrain_models/bert_pytorch_version/bert-base-chinese/')# bert预训练模型

# /home/data1/ftpdata/pretrain_models/OpenCLaP/ms
# BERT_PRETRAINED_PATH = Path('/home/data1/ftpdata/pretrain_models/OpenCLaP/ms/')  # 尝试用法律领域的BERT,在增强数据上效果反而不好
# BERT_PRETRAINED_PATH = Path('../../complaints/bert/pretrained-weights/cased_L-12_H-768_A-12/')
# BERT_PRETRAINED_PATH = Path('../../complaints/bert/pretrained-weights/uncased_L-24_H-1024_A-16/')


# BERT_FINETUNED_WEIGHTS = Path('../trained_model/toxic_comments')

RESULT_PYTORCH_PRETRAINED_BERT = Path('./merge_processed_aug_train_test')  # models   models_sharedatat_tf
RESULT_PYTORCH_PRETRAINED_BERT.mkdir(exist_ok=True)

args = {
    "train_size": -1,
    "val_size": -1,
    "full_data_dir": DATA_PATH,
    "task_name": "toxic_multilabel",
    "no_cuda": False,
    "bert_model": BERT_PRETRAINED_PATH,
    "output_dir": PATH,
    "max_seq_length": 512,#先看下，最长的句子是多长，可以改小，从而batch size可以设置更大
    "do_train": True,
    "do_eval": True,
    "do_lower_case": False,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "learning_rate": init_args.learning_rate,
    "num_train_epochs": 3.0,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "num_gpus": 4,#控制使用单GPU,方便调试
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
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)  # 似乎可以换其他的？
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)  # 在BERT后面接一个dropout，再接一个分类器
        logits = self.classifier(pooled_output)  # 是否使用

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()  # 在使用0和1方式判断该类别是否出现。其中带有sigmoid层了。
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

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


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

    def get_train_examples(self, data_dir, size=-1):
        # 数据的组织方式为csv,第1列为每个句子的id,第2列为句子的本文内容,后续其余列为labels
        if is_use_aug_data:
            # 使用增强数据
            filename = "train_aug.csv"
        else:
            if is_do_model_eval:
                filename = 'train_spilt_0.9.csv'  # 训练集  train_spilt_0.9.csv   train_aug.csv   train_spilt_0.9.csv
            else:
                if data_prefix:
                    # 不需要dev data
                    filename = "{}_train_test.csv".format(data_prefix)
                else:
                    filename = "train.csv"  # 没有从train data中分出dev data
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
        if size == -1:
            print(os.path.join(data_dir, filename))
            data_df = pd.read_csv(os.path.join(data_dir, filename), delimiter="\t")
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df, "train")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename))
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df.sample(size), "train")

    def get_dev_examples(self, data_dir, size=-1):
        """See base class."""
        filename = 'dev_spilt_0.9.csv'  # dev_spilt_0.9.csv   dev_aug.csv   dev_spilt_0.9.csv
        if size == -1:
            data_df = pd.read_csv(os.path.join(data_dir, filename), delimiter="\t")
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df, "dev")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename), delimiter="\t")
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df.sample(size), "dev")

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        data_df = pd.read_csv(os.path.join(data_dir, data_file_name), delimiter="\t")
        #         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
        if size == -1:
            return self._create_examples(data_df, "test")
        else:
            return self._create_examples(data_df.sample(size), "test")

    def get_labels(self):
        """See base class."""
        if self.labels == None:
            self.labels = list(pd.read_csv(os.path.join(self.data_dir, "classes.txt"), header=None)[0].values)
        return self.labels

    def _create_examples(self, df, set_type, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.values):
            guid = row[0]
            text_a = row[1]
            if labels_available:
                labels = row[2:]
            else:
                labels = []
            examples.append(
                InputExample(guid=guid, text_a=text_a, labels=labels))
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
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
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

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
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


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_thresh(y_pred: Tensor, y_true: Tensor, thresh: float = 0.5, sigmoid: bool = True):
    # 为精度度量函数增加了一个阈值，默认设置为0.5。
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()
    #     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    return np.mean(((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred: Tensor, y_true: Tensor, thresh: float = 0.2, beta: float = 2, eps: float = 1e-9,
          sigmoid: bool = True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean().item()


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


processors = {
    "toxic_multilabel": MultiLabelTextProcessor
}

# Setup GPU parameters
if args["num_gpus"] == 1:
    device = torch.device("cuda:2")
    n_gpu = 1
elif args["num_gpus"] > 1:
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"#手动指定GPU
    if args["local_rank"] == -1 or args["no_cuda"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args['local_rank'])
        device = torch.device("cuda", args['local_rank'])
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')


logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args['local_rank'] != -1), args['fp16']))


args['train_batch_size'] = int(args['train_batch_size'] / args['gradient_accumulation_steps'])
random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
if n_gpu > 0:
    torch.cuda.manual_seed_all(args['seed'])

task_name = args['task_name'].lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name](args['full_data_dir'])
# label_list = processor.get_labels()#从tag文件中读取
label_list = tiles
num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])
train_examples = None
num_train_steps = None
if args['do_train']:
    train_examples = processor.get_train_examples(args['full_data_dir'], size=args['train_size'])
    #     train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
    num_train_steps = int(
        len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])


# Prepare model
def get_model():
    #     pdb.set_trace()
    if model_state_dict:
        model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels,
                                                                        state_dict=model_state_dict)
    else:
        model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels)
    return model


model = get_model()

if args['fp16']:
    model.half()

model.to(device)


if args['local_rank'] != -1:
    #使用多个GPU，所以将Pytorch模型封装在DataParallel模块中，这使其能够在所有可用的GPU上进行训练。
    # 没有使用半精度FP16技术，因为使用logits损失函数的二进制交叉熵不支持FP16处理。但这并不会影响最终结果，只是需要更长的时间训练。
    # try:
    #     from apex.parallel import DistributedDataParallel as DDP
    # except ImportError:
    #     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #
    # model = DDP(model)
    pass
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

from torch.optim.lr_scheduler import _LRScheduler, Optimizer


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        #         if not isinstance(optimizer, Optimizer):
        #             raise TypeError('{} is not an Optimizer'.format(
        #                 type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
t_total = num_train_steps
if args['local_rank'] != -1:
    t_total = t_total // torch.distributed.get_world_size()

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args['learning_rate'],
                     warmup=args['warmup_proportion'],
                     t_total=t_total)

scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)


# Eval Fn
def model_eval():
    """
    以验证集对模型进行评估
    :return:
    """
    args['output_dir'].mkdir(exist_ok=True)

    eval_examples = processor.get_dev_examples(args['full_data_dir'], size=args['val_size'])
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args['max_seq_length'], tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    all_logits = None
    all_labels = None

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        #         logits = logits.detach().cpu().numpy()
        #         label_ids = label_ids.to('cpu').numpy()
        #         tmp_eval_accuracy = accuracy(logits, label_ids)
        tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    # ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              #               'loss': tr_loss/nb_tr_steps,
              'roc_auc': roc_auc}

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return eval_loss
    # return result


# load training data
train_features = convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info(" Num train_features = %d", len(train_features))
logger.info("  Batch size = %d", args['train_batch_size'])
logger.info("  Num steps = %d", num_train_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
if args['local_rank'] == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm


def start_train(train_lines, num_epocs=args['num_train_epochs']):
    global_step = 0
    model.train()
    best_loss = 100000.0
    log_step = int(train_lines / args['train_batch_size'] / 4)# 每个epoch验证几次，默认4次
    logger.info("log_step={}".format(log_step))
    # pdb.set_trace()
    for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):
        save_nums = 0  # 每个epoch 中被save的次数
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # logger.info("step={}".format(step))
            # step 是整个数据集分为多个batch之后的顺序标识
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                #             scheduler.batch_step()
                # modify learning rate with special warm up BERT uses
                lr_this_step = args['learning_rate'] * warmup_linear(global_step / t_total, args['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            # 在评估数据增强效果的时候，先不用一个epoch
            # 可以指定 step 进行一次评估，而不必是非要一个epoch评估一次
            # 这里的step整个数据集
            if is_do_model_eval and step % log_step == 4 and i_ > 1:#第3个epoch才需要
                eval_loss = model_eval()  # 使用模型校验集，似乎可以不用
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    # 从第2个epoch开始，保留每个epoch中最好的结果，而不是保留epoch中最后的结果
                    if is_use_aug_data:
                        cur_epoch_best_model = "{}_aug_{}_learnrate_{}_numepoch_{}.bin".format(task_type_name,
                                                                                               model_name,
                                                                                               args['learning_rate'],
                                                                                               i_ + 1)
                    else:
                        cur_epoch_best_model = "{}_{}_learnrate_{}_numepoch_{}.bin".format(task_type_name,
                                                                                           model_name,
                                                                                           args['learning_rate'],
                                                                                           i_ + 1)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    torch.save(model_to_save.state_dict(),
                               os.path.join(RESULT_PYTORCH_PRETRAINED_BERT, cur_epoch_best_model))
                    save_nums = save_nums + 1
                model.train()

        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        logger.info('Eval after epoc {}'.format(i_ + 1))


if n_gpu > 1:
    model.module.unfreeze_bert_encoder()# 多GPU训练
else:
    model.unfreeze_bert_encoder()#单GPU训练

train_features_len = len(train_features)
start_train(train_features_len)#开始训练

# Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
# "{}_{}_learnrate_{}.bin".format(task_type_name, model_name, args['learning_rate'])
if is_use_aug_data:
    result_filename = "{}_aug_{}_learnrate_{}.bin".format(task_type_name, model_name, args['learning_rate'])
else:
    if data_prefix:
        result_filename = "{}_{}_{}_learnrate_{}.bin".format(task_type_name, data_prefix, model_name, args['learning_rate'])
    else:
        result_filename = "{}_{}_learnrate_{}.bin".format(task_type_name, model_name, args['learning_rate'])
output_model_file = os.path.join(RESULT_PYTORCH_PRETRAINED_BERT, result_filename)
torch.save(model_to_save.state_dict(), output_model_file)


# # Load a trained model that you have fine-tuned
# model_state_dict = torch.load(output_model_file)
# model = BertForMultiLabelSequenceClassification.from_pretrained(RESULT_PYTORCH_PRETRAINED_BERT, num_labels=num_labels,
#                                                                 state_dict=model_state_dict)
# model.to(device)
# # model
# model_eval()


# do prediction
# def predict(model, path, test_filename='test.csv'):
#     predict_processor = MultiLabelTextProcessor(path)
#     test_examples = predict_processor.get_test_examples(path, test_filename, size=-1)
#
#     # Hold input data for returning it
#     input_data = [{'id': input_example.guid, 'comment_text': input_example.text_a} for input_example in test_examples]
#
#     test_features = convert_examples_to_features(
#         test_examples, label_list, args['max_seq_length'], tokenizer)
#
#     logger.info("***** Running prediction *****")
#     logger.info("  Num examples = %d", len(test_examples))
#     logger.info("  Batch size = %d", args['eval_batch_size'])
#
#     all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
#     all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
#
#     test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
#
#     # Run prediction for full data
#     test_sampler = SequentialSampler(test_data)
#     test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])
#
#     all_logits = None
#
#     model.eval()
#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_steps, nb_eval_examples = 0, 0
#     for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
#         input_ids, input_mask, segment_ids = batch
#         input_ids = input_ids.to(device)
#         input_mask = input_mask.to(device)
#         segment_ids = segment_ids.to(device)
#
#         with torch.no_grad():
#             logits = model(input_ids, segment_ids, input_mask)
#             logits = logits.sigmoid()
#
#         if all_logits is None:
#             all_logits = logits.detach().cpu().numpy()
#         else:
#             all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
#
#         nb_eval_examples += input_ids.size(0)
#         nb_eval_steps += 1
#
#     return pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns=label_list), left_index=True,
#                     right_index=True)

# print("Start do prediction")
# # 读取test.csv进行测试，划分10%的数据作为测试？？？
# result = predict(model, DATA_PATH)
# print("result.shape=", result.shape)
# result[tiles].to_csv(DATA_PATH/'submission_14_single.csv', index=None)

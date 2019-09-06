# -*- coding: utf-8 -*-
# @CreatTime    : 2019/8/26 14:35
# @Author  : JasonLiu
# @FileName: data_process_config.py
import os


#--------预处理的配置信息--------------
do_raw_text_process = True

#  数据存放位置信息
task_type_name = "labor"  # 需要对3种数据进行训练： loan ， divorce ， labor
num_labels = 20

# 原始数据
stage_1_path = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/data/CAIL2019-FE-Small/"
stage_2_path = "/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/data/CAIL2019-FE-big/"

raw_data_base_dir = os.path.join(stage_2_path, task_type_name)
raw_data_filename = "train_selected.json"  # 对于labor数据集建议采用修正过的数据，因为原始的数据有些垃圾信息。
tag_file = os.path.join(raw_data_base_dir, "tags.txt")

# 基于原始数据生成的中间数据，是否要生成自己目录？
data_base_dir = raw_data_base_dir  # 直接生成在raw data的目录中

is_use_all_train_data = False  # 没有dev data，直接加载train.csv进行训练，不再划分出dev data
is_use_aug = True  # 训练时候是否使用增强后的数据集
if is_use_aug:
    train_file_name = "train_aug.csv"  # train_aug.csv   train.csv
    post_name = "aug"
else:
    train_file_name = "train.csv"
# dev_file_name = "dev.csv" # dev set直接从train data中划分出来

test_data_format = "csv"
if test_data_format == "csv":
    test_file_name = "test.csv"
else:
    # 用第一阶段的json
    test_file_name = "data_small_selected.json"  # 值为 data_small_selected.json   或者   test.csv
    # 也可以用test.json



train_val_ratio = 0.9

##use downloaded model, change path accordingly
model_name = "wwm_ext"
if model_name == "bert":
    BERT_BASE_DIR = "/home/data1/ftpdata/pretrain_models/chinese_L-12_H-768_A-12"  # BERT存放目录
elif model_name == "wwm":
    # 利用loan数据集进行微调，找最优的学习率
    BERT_BASE_DIR = "/home/data1/ftpdata/pretrain_models/chinese_bert_wwm/chinese_wwm_L-12_H-768_A-12"
elif model_name == "wwm_ext":
    BERT_BASE_DIR = "/home/data1/ftpdata/pretrain_models/chinese_bert_wwm/chinese_wwm_ext_L-12_H-768_A-12"
elif model_name == "":  # 领域语料预训练的bert
    BERT_BASE_DIR = ""

BERT_VOCAB = os.path.join(BERT_BASE_DIR, 'vocab.txt')
BERT_INIT_CHKPNT = os.path.join(BERT_BASE_DIR, 'bert_model.ckpt')
BERT_CONFIG = os.path.join(BERT_BASE_DIR, 'bert_config.json')
# We'll set sequences to be at most 128 tokens long.
max_seq_length = 512  # 可以预先评估下每个样本的长度

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
train_batch_size = 4
learning_rate = 4e-5
num_train_epochs = 3.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
warmup_proportion = 0.1
# Model configs
save_checkpoints_steps = 1000
save_summary_steps = 500

if is_use_aug:
    output_dir = os.path.join("./outputs", "{}_aug_{}_learate_{}".format(task_type_name, model_name, learning_rate))
    models_dir = os.path.join("./models", "{}_aug_{}_learate_{}".format(task_type_name, model_name, learning_rate))
else:
    output_dir = os.path.join("./outputs", "{}_{}_learate_{}".format(task_type_name, model_name, learning_rate))
    models_dir = os.path.join("./models", "{}_{}_learate_{}".format(task_type_name, model_name, learning_rate))

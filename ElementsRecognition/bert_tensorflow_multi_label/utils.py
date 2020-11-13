# -*- coding: utf-8 -*-
# @CreatTime    : 2019/8/17 16:52
# @Author  : JasonLiu
# @FileName: utils.py
import tensorflow as tf
import collections
import pandas as pd
import json
import numpy as np
import os
import jieba
import configuration as cfig
from sklearn.model_selection import train_test_split
from bert import optimization
from bert import modeling
import pdb

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids,
        self.is_real_example = is_real_example


def create_examples(df, labels_available=True, dt_format="csv", tag_dic=""):
    """Creates examples for the training and dev sets.
    支持df格式和json格式。注意：线上采用json格式
    """
    examples = []
    if dt_format == "csv":
        for (i, row) in enumerate(df.values):
            guid = row[0]
            text_a = row[1]
            if labels_available:
                labels = row[2:]  # 此时labels类型是array
                # 注意这里的labels的值为0和1
            else:
                #  到底是用空[]还是20*[0]？？
                labels = 20*[0]
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
    else:
        # json格式
        # data_fin = open(os.path.join(data_dir, data_file_name), 'r', encoding='utf8')
        line_data = df.readline()
        line_num = 1
        while line_data:
            d = json.loads(line_data)
            for sent in d:
                sentence = sent['sentence']
                line_num = line_num + 1
                guid = line_num
                text_a = sentence
                if labels_available:
                    temp_labels = sent['labels']  # LB1 - LB20
                    taglist = [0] * 20
                    if temp_labels:
                        # 转为值为0和1,长度为20的list
                        for i in temp_labels:
                            temp = tag_dic[i]
                            taglist[temp] = 1
                else:
                    taglist = []
                examples.append(InputExample(guid=guid, text_a=text_a, labels=taglist))
            line_data = df.readline()
    return examples


def create_examples_text_list(text_list):
    """Creates examples for the training and dev sets.
    方便请求方调用
    """
    examples = []
    labels = 20 * [0]
    for (i, row) in enumerate(text_list):
        guid = i
        text_a = row
        examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
    return examples


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=0,
            is_real_example=False)

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    labels_ids = []
    for label in example.labels:
        labels_ids.append(int(label))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=labels_ids,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        # tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        if isinstance(feature.label_ids, list):
            label_ids = feature.label_ids
        else:
            label_ids = feature.label_ids[0]
        features["label_ids"] = create_int_feature(label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([20], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        # probabilities = tf.nn.softmax(logits, axis=-1) ### multiclass case
        probabilities = tf.nn.sigmoid(logits)  #### multi-label case
        # 在简单的二进制分类中，两者之间没有太大的区别，但是在多分类的情况下，sigmoid允许处理非独占标签（也称为多标签），而softmax处理独占类。

        labels = tf.cast(labels, tf.float32)
        tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        # 用tf.nn.sigmoid_cross_entropy_with_logits测量离散分类任务中的概率误差，其中每个类是独立的而不是互斥的。这适用于多标签分类问题
        loss = tf.reduce_mean(per_example_loss)

        # probabilities = tf.nn.softmax(logits, axis=-1)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #
        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        #
        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):

                logits_split = tf.split(probabilities, num_labels, axis=-1)
                label_ids_split = tf.split(label_ids, num_labels, axis=-1)
                # metrics change to auc of every class
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                    eval_dict[str(j)] = (current_auc, update_op_auc)
                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                return eval_dict

                ## original eval metrics
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # accuracy = tf.metrics.accuracy(
                #     labels=label_ids, predictions=predictions, weights=is_real_example)
                # loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                # return {
                #     "eval_accuracy": accuracy,
                #     "eval_loss": loss,
                # }

            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            print("mode:", mode, "probabilities:", probabilities)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold=scaffold_fn)
        return output_spec

    return model_fn


def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples, cfig.num_labels], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def text_based_input_fn_builder(text_list, seq_length, tokenizer, input_file,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to serving."""
    predict_examples = create_examples_text_list(text_list)
    for (ex_index, example) in enumerate(predict_examples):
        feature = convert_single_example(ex_index, example,
                                         cfig.max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        if isinstance(feature.label_ids, list):
            label_ids = feature.label_ids
        else:
            label_ids = feature.label_ids[0]
        features["label_ids"] = create_int_feature(label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([20], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def serving_input_receiver_fn():
    feature_spec = {
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([cfig.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([cfig.max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([cfig.max_seq_length], tf.int64),
    }
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=None,
                                           name='input_example_tensor')

    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    # tf.Examples
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def serving_input_fn():
    # 保存模型为SaveModel格式
    # 采用最原始的feature方式，输入是feature Tensors。
    # 如果采用build_parsing_serving_input_receiver_fn，则输入是tf.Examples
    label_ids = tf.placeholder(tf.int32, [None, 20], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, cfig.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, cfig.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, cfig.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn


def serving_input_fn_v2():
    # 保存模型为SaveModel格式
    # 如果采用build_parsing_serving_input_receiver_fn，则输入是tf.Examples
    feature_spec = {
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([cfig.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([cfig.max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([cfig.max_seq_length], tf.int64),
    }
    input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()
    return input_fn


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        #print(example.text_a)
        tokens_a = tokenizer.tokenize(str(example.text_a))

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

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
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
            labels_ids.append(int(label))

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


def create_output(predictions, label_columns):
    probabilities = []
    for (i, prediction) in enumerate(predictions):
        preds = prediction["probabilities"]
        probabilities.append(preds)
    dff = pd.DataFrame(probabilities)
    dff.columns = label_columns

    return dff, probabilities


def ensemble_create_output(predictions, predictions_1, label_columns):
    probabilities = []
    for (i, prediction) in enumerate(predictions):
        preds = prediction["probabilities"]
        preds_1 = predictions_1["probabilities"]
        # 进行平均
        probabilities.append(preds)
    dff = pd.DataFrame(probabilities)
    dff.columns = label_columns

    return dff, probabilities


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


stopwords_modify_path = os.path.join(cfig.stop_word_path, cfig.stop_word_filename)
stopwords_modify = open(stopwords_modify_path, mode='r').readlines()
stopwords_modify = [word.strip() for word in stopwords_modify]


def clean_data(name):
    """
    不该把所有标点符号去掉？
    :param name:
    :return:
    """
    setlast = jieba.lcut(name, cut_all=False)
    # stopwords_modify 或者 stopwords
    seg_list = [i.lower() for i in setlast if i not in stopwords_modify]  # 这里的停用词是不一样的，注意
    return "".join(seg_list)


def line_process(linedata, max_len, is_use_stop_words=False):
    # 是否去掉首句，是否使用停用词表？？
    data_str = linedata.replace("\n", "")
    if is_use_stop_words:
        data_str = clean_data(data_str)
    if len(data_str) > max_len:
        max_len = len(data_str)
    return data_str, max_len


def process_data(file_path, tag_dic, titles):
    """
    从原始数据中划分出train data和test data。并以csv格式存储。
    :return:
    """
    fin = open(file_path, 'r', encoding='utf8')
    line = fin.readline()
    sentence_tag_dict = collections.OrderedDict()
    while line:
        d = json.loads(line)
        for sent in d:
            sentence = sent['sentence'].strip()
            labels = sent['labels']  # LB1 - LB20
            if sentence in sentence_tag_dict:
                if sentence_tag_dict[sentence] != labels:
                    #                 print("same sentence={},label_pre={},label_now={}".format(sentence, sentence_tag_dict[sentence], labels))
                    # 对于不一致的情况，选择第一个labels有标注结果的即可。已有的是空，而新的非空，则覆盖。已有的非空，则不替换。
                    if not sentence_tag_dict[sentence]:
                        sentence_tag_dict[sentence] = labels
            else:
                sentence_tag_dict[sent['sentence']] = labels
        line = fin.readline()
    fin.close()

    # 遍历sentence_tag_dict
    alltext = []
    tag_label = []
    processed_max_len = 0
    for sentence in sentence_tag_dict:
        # 在此可以加一个文本预处理操作，比如去掉停用词
        if cfig.do_raw_text_process:
            # 进行文本的预处理
            sentence, processed_max_len = line_process(sentence, processed_max_len, is_use_stop_words=False)
        alltext.append(sentence)
        labels = sentence_tag_dict[sentence]
        taglist = [0] * 20
        if labels:
            for i in labels:
                temp = tag_dic[i]
                taglist[temp] = 1
        tag_label.append(taglist)

    print("len alltext=", len(alltext))  # 未去重前， 5682条
    print("len tag_label=", len(tag_label))
    print("processed_max_len=", processed_max_len)  # 预处理后的文本最大长度
    alltext_set = set(alltext)
    print("去重后的样本数=", len(alltext_set))  # 还是否有重复的呢？

    # 将上述的句子和labels进行拼接
    label_array = np.array(tag_label)
    orign_text_array = np.array(alltext)  # 不带分词
    orign_text_array = orign_text_array.reshape(orign_text_array.shape[0], -1)
    print(orign_text_array.shape)
    print(label_array.shape)
    orign_whole_data_array = np.concatenate([orign_text_array, label_array], axis=1)
    orign_whole_data = pd.DataFrame(orign_whole_data_array, columns=titles)

    if cfig.do_raw_text_process:
        orign_labor_csv_filename = os.path.join(cfig.data_base_dir, "processed_train_test_data.csv")
    else:
        orign_labor_csv_filename = os.path.join(cfig.data_base_dir, "train_test_data.csv")  # 没有test data,用于最后的训练

    orign_whole_data.to_csv(orign_labor_csv_filename, sep="\t", encoding="utf-8",
                            index_label="id")  # index=0,则不保留行索引,header=0则不保存列名

    train_data, test_data = train_test_split(orign_whole_data, random_state=42, train_size=cfig.train_val_ratio,
                                             shuffle=True)
    if cfig.do_raw_text_process:
        train_filename = "processed_train.csv"
        test_filename = "processed_test.csv"
        test_json_file = "processed_test.json"
    else:
        train_filename = "train.csv"
        test_filename = "test.csv"
        test_json_file = "test.json"
    train_path = os.path.join(cfig.data_base_dir, train_filename)  # dev set是从train中再划分出0.1的数据作为dev set
    test_path = os.path.join(cfig.data_base_dir, test_filename)
    train_data.to_csv(train_path, sep="\t", encoding="utf-8", index_label="id")
    test_data.to_csv(test_path, sep="\t", encoding="utf-8", index_label="id")

    # 还需要生成test.json格式，方便与PyTorch版本对比，也方便采用官方的judger.py进行评估
    test_json_file = os.path.join(cfig.data_base_dir, test_json_file)
    test_list = []
    for (i, row) in enumerate(test_data.values):
        sentence = row[1]
        labels = sentence_tag_dict[sentence]
        # 查找对应的label names
        test_dict = {}
        test_dict["sentence"] = sentence
        test_dict["labels"] = labels
        test_list.append(test_dict)
    with open(test_json_file, 'w', encoding="utf-8") as out_test_json_file:
        json.dump(test_list, out_test_json_file, ensure_ascii=False)


def generate_pred_file(tags_list, pred_list, inf_path, outf_path):
    with open(inf_path, 'r', encoding='utf-8') as inf, open(
            outf_path, 'w', encoding='utf-8') as outf:
        line_num = 0
        for line in inf.readlines():
            pre_doc = json.loads(line)
            predict_doc = []
            for ind in range(len(pre_doc)):
                pred_sent = pre_doc[ind]
                # prd此时是一个矩阵
                pred_label = pred_list[line_num]#这里的label是纯数字
                line_num = line_num + 1
                label_names = []
                for label in pred_label:
                    label_names.append(tags_list[label - 1])
                    # pdb.set_trace()
                pred_sent['labels'] = label_names
                predict_doc.append(pred_sent)
            json.dump(predict_doc, outf, ensure_ascii=False)
            outf.write('\n')

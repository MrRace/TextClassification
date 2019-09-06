# CAIL2019--要素识别
本项目用于2019年法研杯要素识别任务的数据、基线模型、传统解决方案和BERT解决方案的整体说明


# 数据说明
- 本任务所使用的数据集是来自“中国裁判文书网”公开的法律文书,数据中的每一行为一篇裁判文书中提取部分段落的分句结果以及句子的要素标签列表；

- 此次比赛主要涉及三个领域：婚姻、劳动争议和借款纠纷

从数据中可以看出，该任务其实是一个多标签分类任务。

# 传统方案
## baseline
- 将下载的训练数据放到data/文件夹下，劳动争议领域训练用的数据在目录labor/ 下，该文件夹下还有少量的测试数据以及svm模型生成的预测数据；另外两个领域的在另外的相应文件夹下；

- baseline文件夹下包括了基于svm的基线模型的训练、预测等相关代码，其中,svm.py包含模型训练的代码，predictor文件夹中包含了数据处理和预测相关代码，修改相应的数据路径后，运行svm.py可以生成模型文件，然后运行predictor.py 可以生成预测文件；基于svm的模型，通过初赛数据的训练，在线上测试集上三个领域的平均得分约为0.4468；

- judger.py 中包含计算模型最终得分的代码，基于micro-f1 和 macro-f1的平均值（该任务最终的得分是三个领域得分的平均值），运行该代码可以输出得分。

## 传统方案
`TraditionalMethods`文件夹中使用的是传统的三种方案：SVM、Xgboost、MLP
`multi_label_models.py`中主要3个参数需要指定。
(1)`clf_type = "xgboost"`  # 支持svm  xgboost   mlp。主要是因为采用`one_vs_one` 或者`one_vs_rest`策略的时候需要遍历所有类别。在处理逻辑上有不同。
(2)`clf_name = "one_vs_rest_xgboost"`  # 指定具体的分类器，这里的管道前置的文字向量化都采用tf-idf，当然也可以自行设计为其他形如Word2Vec的方式。
(3)`is_do_grid_search = True` # 设置是否进行超参数的搜索。


# BERT
## BERT_PyTorch
目录`bert-multilabel`存放的是针对该任务的BERT解决方案(PyTorch版)

(1)模型训练：
`run_train.sh`

(2)进行预测：
`run_predict_judge.sh`


## BERT_TensorFlow
目录`bert_tensorflow_multi_label`存放的是针对该任务的BERT解决方案(TensorFlow版)

(1)配置基本信息：`configuration.py`
(2)运行`multi-label-classification-bert.py`


PS：需要注意的是，预训练的BERT模型只有390MB,但是微调结果模型有1.2GB。这是由于训练过程中的checkpoints由于包含了每个权重变量对应的
`Adam momentum`和`variance`变量，所以训练后的checkpoints是分布式checkpoint的3倍。这多出来的Adam momentum和variance
实际上不是模型的一部分，其作用是能够暂停并在中途恢复训练。
解决方案1：
将模型转为pb格式(`convert_ckpt_to_pb.py`)，然后在预测时候加载pb格式的模型进行推理操作。这里就不过多阐述这个方案。

解决方案2：
直接将模型中多余的变量去掉，实现模型瘦身。具体实现脚本(`smaller_checkpoint_data.py`)如下所示：
```
import tensorflow as tf
meta_file = "./online_models/model.ckpt-20061.meta"
checkpoint_file = "./online_models/model.ckpt-20061"
sess = tf.Session()
imported_meta = tf.train.import_meta_graph(meta_file)
imported_meta.restore(sess, checkpoint_file)
my_vars = []
for var in tf.all_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
        my_vars.append(var)
saver = tf.train.Saver(my_vars)
saver.save(sess, './online_models/divorce_best_model.ckpt')
```

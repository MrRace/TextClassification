# -*- coding: utf-8 -*-
# @CreatTime    : 2019/8/22 19:04
# @Author  : JasonLiu
# @FileName: smaller_checkpoint_data.py

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

#!/bin/bash
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export BERT_BASE_DIR=/home/data1/ftpdata/pretrain_models/chinese_L-12_H-768_A-12
# export GLUE_DIR=/home/liujiepeng/MachineComprehension/CAIL2019/ElementsRecognition/bert_tensorflow_multi_label/bert/data_set
export SIM_DIR=/home/data1/ftpdata/DataResources/Corpus/TextSimilarity/ATEC/normaldata
# /home/data1/ftpdata/DataResources/Corpus/TextSimilarity/ATEC/normaldata
# /home/data1/ftpdata/DataResources/Corpus/TextSimilarity/ATEC/subdata

CUDA_VISIBLE_DEVICES=1 python run_classifier.py \
        --task_name=sim \
        --do_train=true \
        --do_eval=true \
        --data_dir=$SIM_DIR \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=16 \
        --learning_rate=2e-5 \
        --num_train_epochs=3.0 \
        --output_dir=sim_output/
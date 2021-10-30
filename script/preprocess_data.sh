#!/bin/bash

set -x

FILEHOME = music_pre
DATAHOME=data
EXEHOME=src
GLOVEHOME=golve
#cd ${EXEHOME}

python ${EXEHOME}/preprocess.py \
       -train_src ${DATAHOME}/text-data/film_pre/train_src.txt -train_tgt ${DATAHOME}/text-data/film_pre/train_tgt.txt \
       -valid_src ${DATAHOME}/text-data/film_pre/valid_src.txt -valid_tgt ${DATAHOME}/text-data/film_pre/valid_tgt.txt \
       -train_kno ${DATAHOME}/text-data/film_pre/train_graph.txt -valid_kno ${DATAHOME}/text-data/film_pre/valid_graph.txt \
       -train_user ${DATAHOME}/text-data/film_pre/train_src.txt -valid_user ${DATAHOME}/text-data/film_pre/valid_src.txt \
       -train_graph ${DATAHOME}/json-data/film_pre_4child/train_cross.json -valid_graph ${DATAHOME}/json-data/film_pre_4child/valid_cross.json \
       -train_con ${DATAHOME}/text-data/film_pre/train_con.json -valid_con ${DATAHOME}/text-data/film_pre/valid_con.json \
       -save_sequence_data ${DATAHOME}/preprocessed-data/preprcessed_sequence_data_cross_film_pre_4child.pt \
       -save_graph_data ${DATAHOME}/preprocessed-data/preprcessed_graph_data_cross_film_pre_4child.pt \
       -train_dataset ${DATAHOME}/Datasets/train_dataset_cross_film_pre_4child.pt \
       -valid_dataset ${DATAHOME}/Datasets/valid_dataset_cross_film_pre_4child.pt \
       -src_seq_length 50 -tgt_seq_length 50 \
       -src_vocab_size 50000 -tgt_vocab_size 50000 \
       -src_words_min_frequency 3 -tgt_words_min_frequency 2 \
       -vocab_trunc_mode frequency \
       -batch_size 8 \
       -share_vocab \
       -pre_trained_vocab ${GLOVEHOME}/cc.zh.300.vec -word_vec_size 300\
       #-node_feature \
       #-copy \
       #-train_ans ${DATAHOME}/text-data/train.ans.txt -valid_ans ${DATAHOME}/text-data/valid.ans.txt \
       #-answer \
       
       
       

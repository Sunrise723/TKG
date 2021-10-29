 #!/bin/bash

set -x

DATAHOME=data
EXEHOME=src
MODELHOME=models_film_4hop
LOGHOME=log_film_4hop
mkdir -p ${MODELHOME}
mkdir -p ${LOGHOME}

#cd ${EXEHOME}

python src/train.py \
       -sequence_data ${DATAHOME}/preprocessed-data/preprcessed_sequence_data_cross_film_pre_4child.pt \
       -graph_data ${DATAHOME}/preprocessed-data/preprcessed_graph_data_cross_film_pre_4child.pt \
       -train_dataset ${DATAHOME}/Datasets/train_dataset_cross_film_pre_4child.pt \
       -valid_dataset ${DATAHOME}/Datasets/valid_dataset_cross_film_pre_4child.pt \
       -epoch 100 \
       -batch_size 8 -eval_batch_size 16 \
       -training_mode generate \
       -max_token_src_len 200 -max_token_tgt_len 50 \
       -sparse 0 \
       -coverage -coverage_weight 0.4 \
       -d_word_vec 300 \
       -d_seq_enc_model 300 -d_graph_enc_model 300 -n_graph_enc_layer 3 \
       -d_k 300 -brnn -enc_rnn gru \
       -d_dec_model 300 -n_dec_layer 1 -dec_rnn gru \
       -maxout_pool_size 2 -n_warmup_steps 10000 \
       -dropout 0.5 -attn_dropout 0.1 \
       -gpus 1 \
       -save_mode best -save_model ${MODELHOME}/generator \
       -log_home ${LOGHOME} \
       -logfile_train ${LOGHOME}/train_generator \
       -logfile_dev ${LOGHOME}/valid_generator \
       -translate_ppl 80 \
       -curriculum 0  -optim adam \
       -learning_rate 0.00025 -learning_rate_decay 0.75 \
       -valid_steps 120 \
       -decay_steps 120 -start_decay_steps 5000 -decay_bad_cnt 5 \
       -max_grad_norm 5 -max_weight_value 32\
       -pre_trained_vocab\

       #-checkpoint ${MODELHOME}/classifier_cls_84.00753_accuracy.chkpt \
       #-node_feature \
       #-copy \
       #-extra_shuffle 

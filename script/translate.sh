 #!/bin/bash

set -x

DATAHOME=data
EXEHOME=src
MODELHOME=models_duconv
LOGHOME=predictions

mkdir -p ${LOGHOME}

#cd ${EXEHOME}

python src/translate.py \
       -model ${MODELHOME}/generator_grt_3.51009_bleu4.chkpt \
       -sequence_data ${DATAHOME}/preprocessed-data/preprcessed_sequence_data_cross_duconv.pt \
       -graph_data ${DATAHOME}/preprocessed-data/preprcessed_graph_data_cross_duconv.pt \
       -valid_data ${DATAHOME}/Datasets/valid_dataset_cross_duconv.pt \
       -output ${LOGHOME}/prediction_duconv.txt \
       -beam_size 5 \
       -batch_size 4 \
       -gpus 0

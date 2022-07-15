# TKG
This is an implementation of Topic-level Knowledge Sub-Graphs for Multi-turn Dialogue Generation (https://doi.org/10.1016/j.knosys.2021.107499). 
![image](https://github.com/Sunrise723/TKG/blob/main/example.png)

# Requirements
* python3.7
* cuda10.2
* pytorch 1.4.0
* nltk 3.4.4
* numpy 1.18.1
* tqdm 4.32.2
* Memory > 11G (for Knowledge Graphs)

# Data Prepare
To get the topic-level sub-graphs:
```Bash
python data_prepare/datapre.py
python data_prepare/build_tree.py
python data_prepare/build_cross.py
```

# Training & Testing
```Bash
bash scripts/preprocess_data.sh
bash scripts/train_generator.sh
bash scripts/translate.sh
```
# Cite
If the paper is helpful to your research, please kindly cite our paper:
>Jing. Li, Q. Huang, Y. Cai et al., Topic-level knowledge sub-graphs for multi-turn dialogue generation, Knowledge-Based Systems (2021), doi: https://doi.org/10.1016/j.knosys.2021.107499.


#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: eval.py
"""

from __future__ import print_function

import sys
import math
from collections import Counter
from onqg.utils.translate import Translator
import rouge_zh
#import nltk.translate import bleu_score

#reload(sys)
#sys.setdefaultencoding('utf8')


if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " eval_file")
    print("eval file format: pred_response \t gold_response")
    exit()

def dump(filename):
    sent = []
    golds, preds, paras = data[0], data[1], data[2]
    with open(filename, 'w', encoding='utf-8') as f:
        for g, p, pa in zip(golds, preds, paras):
            sent.append([p, g[0]])
    return sent

def get_dict(tokens, ngram, gdict=None):
    """
    get_dict
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None:
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict


def count(pred_tokens, gold_tokens, ngram, result):
    """
    count
    """
    cover_count, total_count = result
    pred_dict = get_dict(pred_tokens, ngram)
    gold_dict = get_dict(gold_tokens, ngram)
    cur_cover_count = 0
    cur_total_count = 0
    for token, freq in pred_dict.items():
        if gold_dict.get(token) is not None:
            gold_freq = gold_dict[token]
            cur_cover_count += min(freq, gold_freq)
        cur_total_count += freq
    result[0] += cur_cover_count
    result[1] += cur_total_count


def calc_bp(pair_list):
    """
    calc_bp
    """
    c_count = 0.0
    r_count = 0.0
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        c_count += len(pred_tokens)
        r_count += len(gold_tokens)
    bp = 1
    if c_count < r_count:
        bp = math.exp(1 - r_count / c_count)
    return bp


def calc_cover_rate(pair_list, ngram):
    """
    calc_cover_rate
    """
    result = [0.0, 0.0] # [cover_count, total_count]
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        count(pred_tokens, gold_tokens, ngram, result)
    cover_rate = (result[0]  + 0.0001) / (result[1] + 0.0001)
    return cover_rate


def calc_bleu(pair_list):
    """
    calc_bleu
    """
    
    bp = calc_bp(pair_list)
    cover_rate1 = calc_cover_rate(pair_list, 1)
    cover_rate2 = calc_cover_rate(pair_list, 2)
    cover_rate3 = calc_cover_rate(pair_list, 3)
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    '''
    weights_f = [0.25, 0.25, 0.25, 0.25]
    weights_th = [0.33, 0.33, 0.33, 0]
    weights_tw = [0.5, 0.5, 0, 0]
    weights_o = [1.0, 0, 0, 0]
        
    bleu_f = bleu_score.corpus_bleu(all_golds, all_preds, weights_f)
    bleu_th = bleu_score.corpus_bleu(all_golds, all_preds, weights_th)
    bleu_tw = bleu_score.corpus_bleu(all_golds, all_preds, weights_tw)
    bleu_o = bleu_score.corpus_bleu(all_golds, all_preds, weights_o)
    return (bleu_f,bleu_th,bleu_te,bleu_o)
    '''
    if cover_rate1 > 0:
        bleu1 = bp * math.exp(math.log(cover_rate1))
    if cover_rate2 > 0:
        bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
    if cover_rate3 > 0:
        bleu3 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3)) / 3)
    return [bleu1, bleu2]
    


def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
        #if freq == 1:
        #    ngram_distinct_count += freq
    return (ngram_distinct_count + 0.0001) / (ngram_total + 0.0001)


def calc_distinct(pair_list):
    """
    calc_distinct
    """
    distinct1 = calc_distinct_ngram(pair_list, 1)
    distinct2 = calc_distinct_ngram(pair_list, 2)
    return [distinct1, distinct2]


def calc_f1(data):
    """
    calc_f1
    """
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    for response, golden_response in data:
        #golden_response = "".join(golden_response).decode("utf8")
        #response = "".join(response).decode("utf8")
        #golden_response = "".join(golden_response)
        #response = "".join(response)
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = hit_char_total / pred_char_total
    r = hit_char_total / golden_char_total
    f1 = 2 * p * r / (p + r)
    return f1

def cacl_rouge(data):
    evaluator_l = rouge_zh.Rouge(metrics=['rouge-l'],
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
    evaluator_n = rouge_zh.Rouge(metrics=['rouge-n'],
                           max_n=4,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
    scores_1 = 0
    scores_2 = 0
    scores_l = 0
    for respo in data:
        respo[0] = " ".join(respo[0])
        respo[1] = " ".join(respo[1])
        scores_1 += evaluator_n.get_scores(respo[0], respo[1])['rouge-1']['r']
        scores_2 += evaluator_n.get_scores(respo[0], respo[1])['rouge-2']['r']
        scores_l += evaluator_l.get_scores(respo[0], respo[1])['rouge-l']['r']
    scores_1 = scores_1/len(data)
    scores_2 = scores_2/len(data)
    scores_l = scores_l/len(data)
    return scores_1, scores_2, scores_l
'''
#if __name__ == "__main__":
   # pass
eval_file = sys.argv[1]
sents = []
for line in open(eval_file):
    tk = line.strip().split("\t")
    if len(tk) < 2:
        continue
    pred_tokens = tk[0].strip().split(" ")
    gold_tokens = tk[1].strip().split(" ")
    sents.append([pred_tokens, gold_tokens])
# calc f1
f1 = calc_f1(sents)
# calc bleu
bleu1, bleu2 = calc_bleu(sents)
# calc distinct
distinct1, distinct2 = calc_distinct(sents)

output_str = "F1: %.2f%%\n" % (f1 * 100)
output_str += "BLEU1: %.3f%%\n" % bleu1
output_str += "BLEU2: %.3f%%\n" % bleu2
output_str += "DISTINCT1: %.3f%%\n" % distinct1
output_str += "DISTINCT2: %.3f%%\n" % distinct2
sys.stdout.write(output_str)
'''
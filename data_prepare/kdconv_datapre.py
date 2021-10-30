import json
import pickle
import argparse
import re
import math

parser = argparse.ArgumentParser()
parser.add_argument('--src_suf', default='duconv_src.txt',
                    help="the suffix of the source filename")
parser.add_argument('--tgt_suf', default='duconv_tgt.txt',
                    help="the suffix of the target filename")
parser.add_argument('--context_suf', default='duconv_con.json',
                    help="the suffix of the context filename")
parser.add_argument('--read_suf', default='duconv.txt',
                    help="the suffix of the target filename")
parser.add_argument('--know_suf', default='duconv_know.txt',
                    help="the suffix of the target filename")
parser.add_argument('--graph', default='duconv_graph.txt',
                    help="the suffix of the target filename")

opt = parser.parse_args(args=[])

maxContextLength = 4


def getdic(dic, dialogA, dialogB, con, knowf, graph):
    dict = eval(dic)
    conversation = dict['conversation']  # tuple
    knowledge = str(dict['knowledge']).replace('\"', '').replace('{', '').replace('}', '').replace('[', '').replace(
        ']', '').replace('\'', '').replace(':', '').replace(',', '')
    index = 0
    context_cu = []
    conversation[1] += conversation[0]
    length = len(conversation[1:-1])
    for i in conversation[1:-1]:
        i = re.sub(r'\[\d\]', "", i)
        context_cu.append(i)
        if length % 2 and index == length - 1:
            break
        if index % 2 == 0:
            con_i = context_cu.copy()
            if len(con_i) > maxContextLength:
                del con_i[0:len(con_i) - maxContextLength]
            con.append(con_i)
            dialogA.write(i + '\n')
        else:
            knowf.write(knowledge + '\n')
            graph.write(dic)
            dialogB.write(i + '\n')
        index += 1
    return con
    # text = text.replace('\n', ',').replace('\r', ',')

def writefile(read_f, dialogAf, dialogBf,contextf, knowf, graphf):
    rfile = open(read_f, 'r')
    dialogA = open(dialogAf, 'w+')
    dialogB = open(dialogBf, 'w+')
    contextfile = open(contextf, 'w+')
    knowledgefile = open(knowf, 'w+')
    graphfile = open(graphf,'w+')
    # 是否需要将对话中的实体嵌入到知识图中？
    # stopfile = open('stopwords.txt' , 'r')
    # stopw = stopfile.read()
    total = 0
    con = []
    for i, dic in enumerate(rfile.readlines()):
        con = getdic(dic, dialogA, dialogB, con, knowledgefile, graphfile)
        
    # print (con)
    final_con = []
    print(len(con))
    final_con.append(con)
    json.dump(final_con, contextfile)
    rfile.close()
    dialogA.close()
    dialogB.close()
    contextfile.close()
    knowledgefile.close()
    graphfile.close()

def main():
    train_src, train_tgt, train_context, train_read, train_know, train_graph = 'train_' + opt.src_suf, 'train_' + opt.tgt_suf, 'train_' + opt.context_suf, \
        'train_' + opt.read_suf, 'train_' + opt.know_suf, 'train_'+opt.graph
    valid_src, valid_tgt, valid_context, valid_read, valid_know, valid_graph = 'valid_' + opt.src_suf, 'valid_' + opt.tgt_suf, 'valid_' + opt.context_suf, \
        'valid_' + opt.read_suf, 'valid_' + opt.know_suf, 'valid_'+opt.graph
    writefile(train_read, train_src, train_tgt, train_context, train_know, train_graph)
    writefile(valid_read, valid_src, valid_tgt, valid_context, valid_know, valid_graph)

    


if __name__ == '__main__':
    main()
import json
import pickle
import argparse
import re
import math

parser = argparse.ArgumentParser()
parser.add_argument('--src_suf', default='src.txt',
                    help="the suffix of the source filename")
parser.add_argument('--tgt_suf', default='tgt.txt',
                    help="the suffix of the target filename")
parser.add_argument('--context_suf', default='con.json',
                    help="the suffix of the context filename")
parser.add_argument('--situation_suf', default='situ.txt',
                    help="the suffix of the situation filename")
parser.add_argument('--goal_suf', default='goal.txt',
                    help="the suffix of the goal filename")
parser.add_argument('--graph_suf', default='graphfile.txt',
                    help="the suffix of the user and knowledge filename")
parser.add_argument('--kno_suf', default='kno.txt',
                    help="the suffix of the knowledge filename")
parser.add_argument('--user_suf', default='user.txt',
                    help="the suffix of the user filename")
parser.add_argument('--read_file', default='train.txt',
                    help="the suffix of the target filename")

opt = parser.parse_args(args=[])

maxContextLength = 4


def getdic(dic, dialogA, dialogB, con, situationfile, goalfile, graphfile, knowledgefile, userfile):
    reword = ' '
    dicts = eval(dic)
    #situation = dicts["situation"]  # str
    #goal = dicts['goal']  # str
    conversation = dicts['conversation']  # tuple
                     = str(dicts['knowledge']).replace('\"', '').replace('{', '').replace('}', '').replace('[', '').replace(
        ']', '').replace('\'', '').replace(':', '').replace(',', '')
    #user = str(dicts['user_profile']).replace('\"', '').replace('{', '').replace('}', '').replace('[', '').replace(
        #']', '').replace('\'', '').replace(':', '').replace(',', '')
    #kno_user = knowledge + " " + user
    index = 0
    context_cu = []
    length = len(conversation)
    for i in conversation:
        '''
        i = i.strip().replace(' ”', '').replace('“ ', '').replace(' … …', '').replace('— — ', '').replace(
            '… ', '').replace('嚯 ', '').replace('镔', '').replace(' `', '').replace('鲅', '').replace(
            '枥', '').replace(' ’', '').replace(' ‘', '').replace('裇 ', '').replace(' 稥', '').replace(
            '诓 ', '').replace('邗', '').replace('苾','').replace('鱻 ','').replace(' ', '').replace(
            '嚯','').replace('唢','').replace(' …','').replace('盍 ','')
        '''
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
            knowledgefile.write(knowledge + '\n')
            #userfile.write(kno_user + '\n')
            #situationfile.write(situation + '\n')
            '''
            goalfile.write(goal + '\n')
            graphfile_dic = dicts.copy()
            graphfile_dic['conversation'] = context_cu
            graphfile_dic['goal'] = [["","",""]]
            graphfile_dic['lable'] = "user"
            graphfile_dic['name'] = dicts['user_profile']['姓名']
            js_graph = json.dumps(graphfile_dic)
            '''
            graphfile.write(dic)
            dialogB.write(i + '\n')
        index += 1
    return con, index
    # text = text.replace('\n', ',').replace('\r', ',')


def divide(readfile, train, valid, test, total):
    rfile = open(readfile, 'r')
    trainfile = open(train, 'w+')
    validfile = open(valid, 'w+')
    testfile = open(test, 'w+')
    num = 0
    print(total)
    trainlen = math.ceil(total / 22 * 20)
    validlen = math.ceil(total / 22 + trainlen)
    print(trainlen)
    print(validlen)
    # import ipdb
    # ipdb.set_trace()
    # print (len(rfile.readlines()))
    a, b, c = 0, 0, 0
    for item in rfile.readlines():
        if num < trainlen:
            trainfile.write(item)
            a += 1
        elif num >= trainlen and num < validlen:
            validfile.write(item)
            b += 1
        elif num >= validlen:
            testfile.write(item)
            c += 1
        num += 1
    print(a, b, c)
    rfile.close()
    trainfile.close()
    validfile.close()
    testfile.close()


def divide_json(readfile, train, valid, test, total):
    rfile = open(readfile, 'r')
    trainfile = open(train, 'w+')
    validfile = open(valid, 'w+')
    testfile = open(test, 'w+')
    confile = json.load(rfile)
    trainlen = math.ceil(total / 22 * 20)
    validlen = math.ceil(total / 22 + trainlen)
    con_train = []
    con_valid = []
    con_test = []
    con_train.append(confile[0][0:trainlen])
    con_valid.append(confile[0][trainlen:validlen])
    con_test.append(confile[0][validlen:])
    json.dump(con_train, trainfile)
    json.dump(con_valid, validfile)
    json.dump(con_test, testfile)
    rfile.close()
    trainfile.close()
    validfile.close()
    testfile.close()


def writefile():
    rfile = open(opt.read_file, 'r')
    dialogA = open(opt.src_suf, 'w+')
    dialogB = open(opt.tgt_suf, 'w+')
    contextfile = open(opt.context_suf, 'w+')
    situationfile = open(opt.situation_suf, 'w+')
    goalfile = open(opt.goal_suf, 'w+')
    graphfile = open(opt.graph_suf, 'w+')
    knowledgefile = open(opt.kno_suf, 'w+')
    userfile = open(opt.user_suf, 'w+')
    # 是否需要将对话中的实体嵌入到知识图中？
    # stopfile = open('stopwords.txt' , 'r')
    # stopw = stopfile.read()
    total = 0
    con = []
    for i, dic in enumerate(rfile.readlines()):
        con, total_cu = getdic(dic, dialogA, dialogB, con, situationfile, goalfile, graphfile, knowledgefile, userfile)
        total += total_cu
    # print (con)
    final_con = []
    final_con.append(con)
    json.dump(final_con, contextfile)
    rfile.close()
    dialogA.close()
    dialogB.close()
    graphfile.close()
    contextfile.close()
    situationfile.close()
    goalfile.close()
    graphfile.close()
    knowledgefile.close()
    userfile.close()
    return total


def main():
    total = writefile() / 2
    train_src, train_tgt, train_graph, train_context, train_situation, train_goal = 'train_' + \
                                                                                    opt.src_suf, 'train_' + opt.tgt_suf, 'train_' + opt.graph_suf, 'train_' + opt.context_suf, \
                                                                                    'train_' + opt.situation_suf, 'train_' + opt.goal_suf,
    valid_src, valid_tgt, valid_graph, valid_context, valid_situation, valid_goal = 'valid_' + \
                                                                                    opt.src_suf, 'valid_' + opt.tgt_suf, 'valid_' + opt.graph_suf, 'valid_' + opt.context_suf, \
                                                                                    'valid_' + opt.situation_suf, 'valid_' + opt.goal_suf,
    test_src, test_tgt, test_graph, test_context, test_situation, test_goal = 'test_' + \
                                                                              opt.src_suf, 'test_' + opt.tgt_suf, 'test_' + opt.graph_suf, 'test_' + opt.context_suf, \
                                                                              'test_' + opt.situation_suf, 'test_' + opt.goal_suf,

    test_user, valid_user, train_user, test_know, valid_know, train_know = 'test_' + \
                                                                           opt.user_suf, 'valid_' + opt.user_suf, 'train_' + opt.user_suf, 'test_' + opt.kno_suf, \
                                                                           'valid_' + opt.kno_suf, 'train_' + opt.kno_suf,

    divide(opt.src_suf, train_src, valid_src, test_src, total)
    divide(opt.tgt_suf, train_tgt, valid_tgt, test_tgt, total)
    #divide(opt.situation_suf, train_situation, valid_situation, test_situation, total)
    #divide(opt.goal_suf, train_goal, valid_goal, test_goal, total)
    print("ssssssssss")
    divide(opt.graph_suf, train_graph, valid_graph, test_graph, total)
    print("bbbbbbbbbbb")
    #divide(opt.user_suf, train_user, valid_user, test_user, total)
    divide(opt.kno_suf, train_know, valid_know, test_know, total)
    divide_json(opt.context_suf, train_context, valid_context, test_context, total)


if __name__ == '__main__':
    main()

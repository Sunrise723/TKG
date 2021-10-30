# -- coding: utf-8 -- 
# knowledge merge to user
from collections import defaultdict
import json
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--graph_suf', default='graphfile.json',
                    help="the suffix of the source filename")
#parser.add_argument('--user_graph_suf', default='user_graphfile.json',
                    #help="the suffix of the source filename")
parser.add_argument('--read_suf', default='graphfile.txt',
                    help="the suffix of the target filename")

opt = parser.parse_args(args=[])

#edge_relation = ['是', '不是', '相同']
edge_relation = ['1', '1', '1']
attr_lable = ['cross','topic']
def getlen(userprofile):
    index = 0
    index += len(userprofile.keys())
    for i in userprofile.values():
        if type(i) == str:
            index += 1
        elif type(i) == list:
            index += len(i)
    return index


def getuser_node_edge(value, dict_len, key_index, value_index, cross, cross_word, is_cross, cross_num):
    node = {}
    edge = [""] * dict_len
    # dep是对边的描述 1是上下位，0是平位，2是拒绝
    # word是结点值

    pat = re.compile(r'没有|拒绝')
    result = pat.findall(value)
    if result:
        node["relation"] = edge_relation[1]
    else:
        node["relation"] = edge_relation[0]
    node["word"] = value
    edge[value_index] = "SELF"
    node["index"] = value_index
    node["child_num"] = 0
    if is_cross:  # 若父节点含有两个以上的孩子
        node["cross"] = cross_num + 1
        node["is_cross"] = cross_word
    elif cross >= 0:  # 若父节点只有一个或0个孩子，且父节点的父节点是一个cross
        node["cross"] = -1
        node["is_cross"] = cross_word
    else:  # 若父节点不是一个cross
        node["cross"] = -1
        node["is_cross"] = "None"

    if key_index == value_index:  # 若键下标和值下标是相等的，该节点是键
        edge[0] = node["relation"]  # user中键挂root is 0
    else:
        edge[key_index] = node["relation"]  # 值挂键

    # index=0 userprofile的键值同root-user相连
    return edge, node


def getuserTree(userprofile):
    # import ipdb
    # ipdb.set_trace()
    nodescontent = []
    edgescontent = []
    diclen = getlen(userprofile) + 1
    rootnode = {"relation": edge_relation[0], "word": "user", "index": 0, "child_num": 0, "cross": 0,
                "is_cross": "None"}
    rootedge = [""] * diclen  # TODO： FIX EDGE RELATION
    rootedge[0] = "SELF"
    nodescontent.append(rootnode)
    edgescontent.append(rootedge)
    keyindex = 0
    valueindex = 0
    cross_num = nodescontent[keyindex]["cross"]
    for i in userprofile:
        cross_word = nodescontent[keyindex]["word"]
        cross = nodescontent[keyindex]["cross"]
        is_cross = False
        keyindex = valueindex + 1
        valueindex = keyindex
        # i 连接user
        if keyindex == valueindex:
            nodescontent[0]["child_num"] += 1
        if type(userprofile[i]) == list and len(userprofile[i]) > 1:
            is_cross = True
            cross_num += 1
        this_edge, this_node = getuser_node_edge(i, diclen, keyindex, valueindex, cross, cross_word, is_cross,
                                                 cross_num)
        edgescontent.append(this_edge)
        nodescontent.append(this_node)
        is_cross = False
        if type(userprofile[i]) == str:
            # userprofile[i]连接 i ，只有一个子节点
            valueindex += 1
            this_edge, this_node = getuser_node_edge(userprofile[i], diclen, keyindex, valueindex, cross, cross_word,
                                                     is_cross, cross_num)
            edgescontent.append(this_edge)
            nodescontent.append(this_node)
        elif type(userprofile[i]) == list:
            # 连接i，有一个或多个子节点
            # is_cross = 1
            if len(userprofile[i]) > 1:
                # 更新keyindex
                cross_word = nodescontent[keyindex]["word"]
                cross = nodescontent[keyindex]["cross"]
            for value in userprofile[i]:
                nodescontent[keyindex]["child_num"] += 1
                valueindex += 1
                this_edge, this_node = getuser_node_edge(value, diclen, keyindex, valueindex, cross, cross_word,
                                                         is_cross, cross_num)
                edgescontent.append(this_edge)
                nodescontent.append(this_node)
    return nodescontent, edgescontent, diclen, cross_num


def getparents(i, k, content, nodescontent, edgescontent):
    # knowledge["聊天"，"日期"，"2019-1-18"]
    dex = -1
    pot = 0
    for c in content:
        # 找dex位置上为1的下标
        # 第一位相似继续往下找
        if c['word'] == k[i]:
            dex = c["index"]
            i += 1
            if i >= len(k): break
            sub_content = [nodescontent[edgescontent.index(edge)] for edge in edgescontent if edge[dex] in edge_relation]
            if sub_content:
                getparents(i, k, sub_content, nodescontent, edgescontent)
            else:
                break
        
    # 返回dex是父节点的下标,i是匹配到的knowledge的下标
    #print(dex, i)
    return dex, i


def getknow_node_edge(dex, value, nodeslen):
    valueindex = nodeslen
    valueindex -= 1
    node = {}
    edge = [""] * (nodeslen + 1)  # 后期需要在后面填充！！！
    #node["relation"] = relation
    node["word"] = value
    node["index"] = nodeslen
    node["child_num"] = 0
    node["cross"] = -1
    #node["is_cross"] = cross_word#TODO cross_word maybe not a cross
    if not dex == -1:
        edge[dex] = "1"
    edge[nodeslen] = "SELF"
    return edge, node


def getknowTree(knowledge):
    # knowledge三元组类型：["聊天"，"日期"，"2019-1-18"]
    # 0位判断是否同usernodes相同，1位判断是否同knowledgenodes相同
    # 先从1位开始判断，再从0位开始判断，
    # nodeslen是user的所有的节点的长度
    #user_nodescontent, user_edgescontent, nodeslen, cross_num = getuserTree(userprofile)
    nodescontent = []
    edgescontent = []
    nodeslen  = 0 
    cross_num = 0
    #listlen = len(knowledge)
    #keyindex = 0
    #length = len(edgescontent[-1])
    #is_similar = -1
    for index_k, k in enumerate(knowledge):
        dex, k_i = getparents(0, k, nodescontent, nodescontent, edgescontent)
        #if dex == -1: continue  # dex是父节点的下标
        if dex != -1:
            if nodescontent[dex]["child_num"] > 1 and nodescontent[dex]["cross"] < 0:#TODO FIX IT!
                cross_num += 1
                nodescontent[dex]["cross"] = cross_num
            cross_word = nodescontent[dex]["word"]
        while k_i < len(k):
            # 更新边结点
            for item in edgescontent:
                item.append("")
            value = k[k_i]
            edge, node = getknow_node_edge(dex, value, nodeslen)
            if dex != -1:
                nodescontent[dex]["child_num"] += 1
            dex = nodeslen
            edgescontent.append(edge)
            nodescontent.append(node)
            k_i += 1
            nodeslen += 1
    return nodescontent, edgescontent


def tree(): return defaultdict(tree)


def main(r, w_know):
    user = []
    know = []
    rfile = open(r, 'r')
    wfile = open(w_know, 'w+')
    #w_user_file = open(w_user, 'w+')
    
    for num, item in enumerate(rfile):

        knowledge = list(item)
        #knowledge = eval(item)['knowledge']  # tuple
        #user_profile = eval(item)['user_profile']  # tuple
        nodescontent, edgescontent = getknowTree(knowledge)
        #users = tree()
        #users['nodes'] = user_nodescontent
        #users['edges'] = user_edgescontent
        knowledge = tree()
        knowledge['nodes'] = nodescontent
        knowledge['edges'] = edgescontent
        #user.append(users)
        know.append(knowledge)
    rfile.close()
    json.dump(know, wfile)
    #json.dump(user, w_user_file)
    wfile.close()


if __name__ == '__main__':
    rtrainfile, rvalidfile, rtestfile = "train_" + opt.read_suf, "valid_" + opt.read_suf, \
                                        "test_" + opt.read_suf
    wtrainfile, wvalidfile, wtestfile = "train_" + opt.graph_suf, "valid_" + opt.graph_suf, \
                                        "test_" + opt.graph_suf
   # w_train_user_file, w_valid_user_file, w_test_user_file = "train_" + opt.user_graph_suf, "valid_" + opt.user_graph_suf, \
                                        #"test_" + opt.user_graph_suf
    main(rtrainfile, wtrainfile)
    main(rvalidfile, wvalidfile)
    #main(rtestfile, wtestfile)

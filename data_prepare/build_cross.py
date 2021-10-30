# -- coding: utf-8 --
# knowledge merge to user
from collections import defaultdict
import json
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--cross_suf', default='cross.json',
                    help="the suffix of the source filename")
parser.add_argument('--read_suf', default='graphfile.json',
                    help="the suffix of the target filename")

opt = parser.parse_args(args=[])

edge_relation = ['是', '不是', '相同']
atttr_relation = ['cross','topic']

def get_member(nodes, edges):
    cross = []
    for i, node in enumerate(nodes):
        member = []
        if node["cross"] >= 0:
            member.append(node["word"])
            member = [nodes[j]["word"] for j, edge in enumerate(edges) if edge[i]]
            cross.append(member)
    return cross

#node and edge form:
'''
[
    ["node":[
        {
            "word":
            "cross":
            "attr":
        },
        {},{}
    ],
    "edge":[[
        "","",""
    ],[],[]],
    []
]
'''

def node_form(word, cross_num, attr, bef_index, index):
    member = {}
    edge = [""] * index
    edge[bef_index] = "1"
    edge[index-1] = "SELF"
    member['word'] = word
    member['cross'] = cross_num
    member['attr'] = attr
    member['index'] = index
    return member, edge

'''
def get_leaf(nodes,edges,cross,cross_edge,cross_num,cross_word,edge_i, bef_index, index, is_child):
    #import ipdb; ipdb.set_trace()
    for j ,edge in enumerate(edges):
        if edge[edge_i] and edge[edge_i] != 'SELF':
            #print(nodes[j])
            #孩子多于一个时，这是一个cross,若这个节点直属于ｃｒｏｓｓ节点，那就是ｔｏｐｉｃ，若是节点的节点，那就是ｃｒｏｓｓ＿ｗｏｒｄ
            #孩子等于一个时，其孩子节点的ｃｒｏｓｓ＿ｗｏｒｄ是前一个
            #孩子少于一个时，没有下文
            if nodes[j]["child_num"] > 1:
                index += 1
                if is_child == False:
                    cross_word = atttr_relation[1]
                node_member, edge_member = node_form(nodes[j]["word"],cross_num,cross_word,bef_index, index)
                cross.append(node_member)
                cross_edge.append(edge_member)
                continue 
            else:
                index += 1
                if is_child == False:#TODO:ｉｓ＿ｃｈｉｌｄ的递归有问题
                    cross_word = atttr_relation[1]
                node_member, edge_member = node_form(nodes[j]["word"], cross_num, cross_word,bef_index, index)
                cross.append(node_member)
                cross_edge.append(edge_member)
                if nodes[j]["child_num"] == 1:
                    is_child = True
                    get_leaf(nodes, edges, cross, cross_edge, cross_num, nodes[j]["word"], j, index-1, index, is_child)
'''              
def get_cross(nodes, edges, cross, cross_edge, cross_num, cross_word, edge_i, bef_index,child_index, index, child_num, child):
    for j ,edge in enumerate(edges):
        if child_index-child >= child_num:
            break
        if edge[edge_i] and edge[edge_i] != 'SELF':
            index += 1
            child_index += 1
            node_member, edge_member = node_form(nodes[j]["word"],cross_num,cross_word,bef_index, index)
            cross.append(node_member)
            cross_edge.append(edge_member)
            if nodes[j]["child_num"] <= 1:
                index = get_cross(nodes, edges, cross, cross_edge, cross_num, nodes[j]["word"], j, \
                    index-1, 0, index, 1, 0)#TODO FIX MAGIC NUMBER
    return index
                


def pad_edge(edge):
    max_length = max([len(i) for i in edge])
    for j in edge:
        j.extend("" for _ in range(max_length- len(j)))

'''
def divide_graph(nodes,edges):
    cross = []
    cross_edge = []
    cross_num = 0
    bef_index = 0
    index = 1
    #member.append(node["word"])
    cross_word = nodes[0]["word"]
    node_member, edge_member = node_form(nodes[0]["word"], cross_num, atttr_relation[0], bef_index, index)
    cross.append(node_member)
    cross_edge.append(edge_member)
    #若有节点同当前ｎｏｄｅ相连，即对应位置的ｅｄｇｅ不为空
    get_leaf(nodes, edges, cross, cross_edge, cross_num, cross_word, 0, bef_index,index)
    pad_edge(cross_edge)
    return cross, cross_edge

'''
def tree(): return defaultdict(tree)

def divide_graph(nodes,edges):
    cross_num = -1
    cross = []
    for i,node in enumerate(nodes):
        if node["cross"] >= 0:#代表这个node含有多个子节点，是一个cross
            cross_node = []
            cross_edge = []
            cross_num += 1
            bef_index = 0
            index = 1
            child_num = node["child_num"]
            child = 0
            node_member, edge_member = node_form(node["word"], cross_num, atttr_relation[0], bef_index, index)
            cross_node.append(node_member)
            cross_edge.append(edge_member)
            #若有节点同当前ｎｏｄｅ相连，即对应位置的ｅｄｇｅ不为空
            get_cross(nodes, edges, cross_node, cross_edge, cross_num, atttr_relation[1], i, bef_index,\
                child, index, child_num, child)
            pad_edge(cross_edge)
            one_cross = tree()
            one_cross['nodes'] = cross_node
            one_cross['edges'] = cross_edge
            cross.append(one_cross)
    if not cross:
        one_cross = tree()
        one_cross['nodes'] = nodes
        one_cross['edges'] = edges
        cross.append(one_cross)
    return cross

def main(r, w):
    with open(r, 'r') as fr:
        r_data = json.load(fr)
    wfile = open(w, 'w+')
    user = []
    
    for num, item in enumerate(r_data):
        nodes = item["nodes"]
        edges = item["edges"]
        cross = divide_graph(nodes, edges)
        
        if not cross:
            print(user[0])
            print(nodes)
            print(edges)
            print(cross)
            print("*****")
        print(len(cross))
        user.append(cross)
        #print("len:",len(user))
    #print("len:",len(user))
    #print(user[0][0])
    print("Cross graph Done")
    json.dump(user, wfile)
    print("Save Done")
    wfile.close()
    fr.close()


if __name__ == '__main__':
    r_train_file, r_valid_file, r_test_file = "train_" + opt.read_suf, "valid_" + opt.read_suf, \
                                              "test_" + opt.read_suf
    w_train_file, w_valid_file, w_test_file = "train_" + opt.cross_suf, "valid_" + opt.cross_suf, \
                                              "test_" + opt.cross_suf
    main(r_train_file, w_train_file)
    main(r_valid_file, w_valid_file)
    #main(r_test_file, w_test_file)

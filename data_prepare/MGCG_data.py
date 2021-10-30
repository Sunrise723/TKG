import json
import pickle
import argparse
import re
import math

parser = argparse.ArgumentParser()
parser.add_argument('--src_suf', default='new_kdconv.txt',
                    help="the suffix of the source filename")
parser.add_argument('--read_suf', default='kdconv.txt',
                    help="the suffix of the target filename")
opt = parser.parse_args(args=[])


def change_form(dic, new_dict_file):
    dict = eval(dic)
    new_dict = {}
    new_dict['goal'] = [["","",""]]
    new_dict['lable'] = "user"
    new_dict['name'] = "姓名＂
    new_dict['conversation'] = dict['conversation']
    new_dict['knowledge'] = dict['knowledge']
    new_dict['situation'] = ""
    write_dict = json.dumps(new_dict)
    new_dict_file.write(write_dict +'\n')


def writefile(read_f, new_dict):
    rfile = open(read_f, 'r')
    new_dict_file = open(new_dict, 'w+')
    for i, dic in enumerate(rfile.readlines()):
        js_dic = json.loads(dic)
        js = json.dumps(js_dic)
        new_dict_file(js,'\n')
    rfile.close()
    new_dict_file.close()

def main():
    train_src, train_read = 'train_' + opt.src_suf,'train_' + opt.read_suf
    valid_src, valid_read = 'valid_' + opt.src_suf,'valid_' + opt.read_suf
    writefile(train_read, train_src)
    writefile(valid_read, valid_src)

    


if __name__ == '__main__':
    main()
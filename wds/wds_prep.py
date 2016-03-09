#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pynlpir
import pickle

from hanzi_util import is_zhs, is_punct

IN_FILE = '结果文件_TAO_T.txt'
YL_FILE = 'v5语料库.txt'
YL_FILE = 'v5语料库_2000k.txt'
YL_FILE = 'v5语料库_5000k.txt'


DICT_FILE = 'dict_dump.dat'

YLP_FILE = YL_FILE + ".p"

PADDING = '_P_'
PUNCING = '_E_'

DICT_W_D = {}
DICT_D_W = {}

def term_to_id(term):
    if term not in DICT_W_D:
        DICT_W_D[term] = len(DICT_D_W)
        DICT_D_W[DICT_W_D[term]] = term
    return DICT_W_D[term]

def get_term_id(term):
    if term not in DICT_W_D:
        return DICT_W_D[PADDING]
    return DICT_W_D[term]

def get_term_wd(id):
    if id not in DICT_D_W:
        return PADDING
    return DICT_D_W[id]

def prep_word_dict():
    
    CURRENT_W = None
    with open(IN_FILE) as fin:
        while True:
            try:
                line = fin.readline()
            except:
                print("READ ERROR:%d" %(LINE_NUM) )
                continue
            if not line:
                print("PROCESS DONE!")
                break

            if line[:4] == '[DDv' :
                CURRENT_W = line[5: line.index(']')]
                term_to_id(CURRENT_W)
                continue

            if CURRENT_W and line[0] == '【' and ('=】' in line):
                line_x = line[line.index('】')+1:]
                line_x = line_x.split()
                if line_x:
                    for item in line_x:
                        term_to_id(item)
                continue

    LINE_NUM = 0
    with open(YL_FILE) as fin, open(YLP_FILE, 'w') as fout:
        while True:
            try:
                line = fin.readline()
            except:
                print("READ ERROR:%d" %(LINE_NUM) )
                continue
            if not line:
                print("PROCESS DONE!")
                break

            LINE_NUM += 1
            if not (LINE_NUM % 5000): print('C:%d' %(LINE_NUM))
            if len(line) > 30: continue

            seg_list = pynlpir.segment(line, pos_tagging=False)
            for i in range(len(seg_list)):
                if is_zhs(seg_list[i]):
                    term_to_id(seg_list[i])
                elif len(seg_list[i]) == 1 and is_punct(seg_list[i]):
                    seg_list[i] = PUNCING
                else:
                    seg_list[i] = PADDING
            fout.write(' '.join(seg_list) + '\n')

    term_to_id(PADDING)
    #term_to_id(PUNCING)
    print('SEN DONE!')


def dump_dicts(dump_file):
    global DICT_W_D
    global DICT_D_W

    print("DUMPING RESULTS...")
    with open(dump_file,'wb', -1) as fp:
        dump_data = []
        dump_data.append(DICT_W_D)
        dump_data.append(DICT_D_W)
        pickle.dump(dump_data, fp, -1)
        del dump_data

    print("字典大小：%d" %(len(DICT_D_W)))
    return

def load_dicts(dump_file):
    global DICT_W_D
    global DICT_D_W

    print("LOADING RESULTS...")
    with open(dump_file,'rb', -1) as fp:
        dump_data = pickle.load(fp)
        DICT_W_D = dump_data[0]
        DICT_D_W = dump_data[1]
        del dump_data    

    print("字典大小：%d" %(len(DICT_D_W)))
    return
    

if __name__ == '__main__':
    
    pynlpir.open()

    prep_word_dict()
    dump_dicts(DICT_FILE)

    print("BEFORE:%d-%d" %(len(DICT_W_D), len(DICT_W_D)))
    DICT_W_D = {}
    DICT_D_W = {}
    print("MIDDLE:%d-%d" %(len(DICT_W_D), len(DICT_W_D)))
    load_dicts(DICT_FILE)
    print("AFTER:%d-%d" %(len(DICT_W_D), len(DICT_W_D)))

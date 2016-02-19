#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import jieba

import os
import sys
import subprocess

IN_FILE = "199801.txt"
TRAIN_FILE = IN_FILE+".train"
TEST_FILE  = IN_FILE+".test"
MODEL_FILE = IN_FILE+".model"

dest_tag = ['nr', 'ns', 'nt', 'nz']

#w[-2]=He, w[-1]=reckons, w[0]=the, w[1]=current, w[2]=account
#w[-1]|w[0]=reckons|the, w[0]|w[1]=the|current

def generate_train_st(line):
    ret_list = []

    if not line:
        return None

    lin = line.split()
    if len(lin) < 3:
        return None

    len_t = len(lin)
    w = []
    t = []
    for i in range(len_t):
        # We currently not care about [ xxx xx ]nz case
        tt = lin[i].split('/')
        if tt[0][0] == '[' and len(tt[0]) > 1:
            tt[0] = tt[0][1:]
        if ']' in tt[1] and tt[1][-3:] == ']nz':
            tt[1] = tt[1][:-3]
        w.append(tt[0])
        t.append(tt[1])

    for i in range(len_t):
        str = t[i] + "\t"

        # USING '*' for patching
        if (i-2) >= 0:
            str += "w[-2]=" + w[i-2] + '\t'
        else:
            str += "w[-2]=" + '*' + '\t'

        if (i-1) >= 0:
            str += "w[-1]=" + w[i-1] + '\t'
            str += "w[-1]|w[0]=" + w[i-1] + '|' + w[i] + '\t'
        else:
            str += "w[-1]=" + '*' + '\t'
            str += "w[-1]|w[0]=" + '*' + '|' + w[i] + '\t'

        str += "w[0]=" + w[i] + '\t'

        if (i+1) < len_t:
            str += "w[1]=" + w[i+1] + '\t'
            str += "w[0]|w[1]=" + w[i] + '|' + w[i+1] + '\t'
        else:
            str += "w[1]=" + '*' + '\t'
            str += "w[0]|w[1]=" + w[i] + '|' + '*' + '\t'

        if (i+2) < len_t:
            str += "w[2]=" + w[i+2] + '\t'
        else:
            str += "w[2]=" + '*' + '\t'
        
        str += "\n"

        ret_list.append(str)

    return ret_list

def generate_test_st(str_list):
    ret_list = []

    if not str_list:
        return None

    if len(str_list) < 3:
        return None

    len_t = len(str_list)
    w = str_list
    for i in range(len_t):
        str = "ZZZ" + "\t"

        # USING '*' for patching
        if (i-2) >= 0:
            str += "w[-2]=" + w[i-2] + '\t'
        else:
            str += "w[-2]=" + '*' + '\t'

        if (i-1) >= 0:
            str += "w[-1]=" + w[i-1] + '\t'
            str += "w[-1]|w[0]=" + w[i-1] + '|' + w[i] + '\t'
        else:
            str += "w[-1]=" + '*' + '\t'
            str += "w[-1]|w[0]=" + '*' + '|' + w[i] + '\t'

        str += "w[0]=" + w[i] + '\t'

        if (i+1) < len_t:
            str += "w[1]=" + w[i+1] + '\t'
            str += "w[0]|w[1]=" + w[i] + '|' + w[i+1] + '\t'
        else:
            str += "w[1]=" + '*' + '\t'
            str += "w[0]|w[1]=" + w[i] + '|' + '*' + '\t'

        if (i+2) < len_t:
            str += "w[2]=" + w[i+2] + '\t'
        else:
            str += "w[2]=" + '*' + '\t'
        
        str += "\n"

        ret_list.append(str)

    return ret_list

def train_model():
    LINE_NU = 0
    with open(IN_FILE) as fin, open(TRAIN_FILE, "w") as ftrain, open(EVAL_FILE, "w") as feval:
        while True:
            try:
                line = fin.readline()
            except:
                print("READ ERROR:%d"%(LINE_NU))

            if not line:
                print("PROCESS DONE!")
                break

            line = line.strip()

            ret_list = generate_train_st(line)
            if ret_list:
                LINE_NU += 1
                for item in ret_list: ftrain.write(item)
    
    print("TRAIN THE MODEL...")
    #CMD = "./crfsuite learn -p max_iterations=20 -m %s %s" %(MODEL_FILE, TRAIN_FILE)
    CMD = "./crfsuite learn -p max_iterations=500 -a lbfgs -p c1=1 -p c2=0 -m %s %s" %(MODEL_FILE, TRAIN_FILE)
    print("CMD:%s"%(CMD))
    os.system(CMD)

def tag_me(str_list):
    ret_list = generate_test_st(str_list)
    if not ret_list: return None

    with open(TEST_FILE,'w') as fout:
        for item in ret_list:
            fout.write(item)
    CMD = "./crfsuite tag -m %s %s" %(MODEL_FILE, TEST_FILE)
    print("CMD:%s"%(CMD))
    p = subprocess.Popen(CMD.split(), stdout=subprocess.PIPE)
    (stdout, stderr) = p.communicate()

    if p.returncode != 0:
        print(stderr)
        raise OSError('%s command failed!' %(CMD))

    return stdout.decode('utf-8').strip().split('\n')

if __name__ == "__main__":

    if not os.path.exists(MODEL_FILE):
        train_model()

    # 因为模型的效果跟分词十分相关，所以必须保证跟标注数据分词一致
    str = "国家 主席 江 泽民 表示 [香港 特别 行政区 红十字会]、[中国 红十字会]为地震灾区提供的援助表示感谢,对于南京,武汉、成都等地区"

    seg_list = list(jieba.cut(str, cut_all=False))
    while ' ' in seg_list:
        seg_list.remove(' ')

    result = tag_me(seg_list)
    for i in range(len(result)):
        if result[i] in dest_tag:
            print("%s\t%s"%(seg_list[i], result[i]))
        else:
            print("%s"%(seg_list[i]))

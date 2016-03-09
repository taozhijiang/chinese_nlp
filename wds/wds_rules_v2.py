#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import re
import pynlpir
import pickle
import os

from wds_prep import load_dicts, get_term_id, get_term_wd, PADDING, PUNCING, DICT_FILE

IN_FILE = '结果文件_TAO_T.txt'
YL_FILE = 'v5语料库.txt'
YL_FILE = 'v5语料库_2000k.txt'
YL_FILE = 'v5语料库_5000k.txt'
YLP_FILE = YL_FILE + ".p"

MODEL_FILE = "model_dump.dat"

TAG_SHIFT = 32
TAG_MASK  = 0xffffffff

# 格式： 

RESULTS_WS = {} #多义词词汇
RESULTS_AIM = [] #需要监测的词表
RESULTS_FQ = {} #词频统计
RESULTS_EX = {} #多义词例句
RESULTS_STOP = []

global DETECT_ERR

def append_to_fq(c_id, w_id_p, w_id_n):
    global RESULTS_FQ
    
    if w_id_p == w_id_n and w_id_p == get_term_id(PADDING):
        return

    tmp_id = w_id_p << TAG_SHIFT | w_id_n

    if tmp_id not in RESULTS_FQ:
        RESULTS_FQ[tmp_id] = {}
        RESULTS_FQ[tmp_id][c_id] = 1
    else:
        if c_id not in RESULTS_FQ[tmp_id]:
            RESULTS_FQ[tmp_id][c_id] = 1
        else:
            RESULTS_FQ[tmp_id][c_id] += 1
    return
    

# 处理语料库，统计词共现环境
def collect_env():
    global RESULTS_AIM
    global RESULTS_FQ
    
    RESULTS_FQ = {}

    LINE_NUM = 0

    with open(YLP_FILE) as fin:
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

            line_x = line.split()
            len_x  = len(line_x)

            for i in range(len_x):
                if line_x[i] in RESULTS_AIM:
                    c_id = get_term_id(line_x[i])                     
                    if (i-1) >= 0:    w_id_p =  get_term_id(line_x[i-1])
                    else:             w_id_p =  get_term_id(PADDING)
                    if (i+1) < len_x: w_id_n =  get_term_id(line_x[i+1])
                    else:             w_id_n =  get_term_id(PADDING)
                    append_to_fq(c_id, w_id_p, w_id_n)

                    if (i-2) >= 0:
                        w_id_p =  get_term_id(line_x[i-2])
                        w_id_n =  get_term_id(line_x[i-1])
                        append_to_fq(c_id, w_id_p, w_id_n)

                    if (i+2) < len_x:
                        w_id_p =  get_term_id(line_x[i+1])
                        w_id_n =  get_term_id(line_x[i+2])
                        append_to_fq(c_id, w_id_p, w_id_n)

    return

# 处理同义词词林，进行词汇的整理    
def process_tyccl():
    global DETECT_ERR
    global RESULTS_WS
    global RESULTS_AIM
    global RESULTS_EX

    RESULTS_WS = {}
    RESULTS_AIM = []
    CURRENT_W = None
    CURRENT_L = None

    len_lj = len('#例句:')

    with open(IN_FILE) as fin:
        while True:
            try:
                line = fin.readline()
            except:
                print("READ ERROR:%d"%(LINE_NU))

            if not line:
                print("PROCESS DONE!")
                break

            line = line.strip()

            if not line :  
                CURRENT_L = None
                continue

            # 新词处理的开始
            if line == '#------------------------------------':
                CURRENT_W = None
                CURRENT_L = None
                continue

            if line[:4] == '[DDv' :
                CURRENT_W = line[5: line.index(']')]
                if CURRENT_W not in RESULTS_WS:
                    RESULTS_WS[CURRENT_W] = {}
                    RESULTS_EX[CURRENT_W] = {}
                continue

            if CURRENT_W and line[0] == '【' and ('=】' in line):
                CURRENT_L = line[: line.index('】')+1]
                line_x = line[line.index('】')+1:]
                line_x = line_x.split()
                if line_x:
                    RESULTS_WS[CURRENT_W][CURRENT_L] = line_x
                    RESULTS_AIM.extend(line_x)
                else:
                    DETECT_ERR = True
                    print('ERROR!!!!!')
                continue

            if line and (not CURRENT_L) or (not CURRENT_W):
                continue

            # # 处理例句部分
            # target = ' %s/%s ' %(CURRENT_W, CURRENT_L)
            # line_aa = []
            # if line[:len_lj] == '#例句:':
            #     line = line[len_lj:]
            #     len_t = len(re.findall(CURRENT_W, line))
            #     if len_t == 0:
            #         continue
            #     elif len_t == 1: #replace
            #         line = line.replace(CURRENT_W,target)
            #     else:
            #         #seg_list = list(jieba.cut(line, cut_all=False))
            #         seg_list = pynlpir.segment(line, pos_tagging=False)
            #         if CURRENT_W not in seg_list:
            #             print("GIVEUP:" + CURRENT_W + "xxx" +' '.join(seg_list))
            #             continue # 放弃
            #         else:
            #             for i in range(len(seg_list)):
            #                 if seg_list[i] == CURRENT_W:
            #                     seg_list[i] = target
            #             line = ''.join(seg_list)
            #     line_aa = [line]
            # elif '～' in line:
            #     line = line.replace('～',target)
            #     line_aa = re.split('丨|│|ㄧ', line)
            # else:
            #     print(line)

            # 处理例句部分
            target = CURRENT_W
            line_aa = []
            if line[:len_lj] == '#例句:':
                line = line[len_lj:]
                len_t = len(re.findall(CURRENT_W, line))
                if len_t == 0:
                    continue
                elif len_t == 1: #replace
                    line = line.replace(CURRENT_W,target)
                else:
                    #seg_list = list(jieba.cut(line, cut_all=False))
                    seg_list = pynlpir.segment(line, pos_tagging=False)
                    if CURRENT_W not in seg_list:
                        print("GIVEUP:" + CURRENT_W + "xxx" +' '.join(seg_list))
                        continue # 放弃
                    else:
                        for i in range(len(seg_list)):
                            if seg_list[i] == CURRENT_W:
                                seg_list[i] = target
                        line = ''.join(seg_list)
                line_aa = [line]
            elif '～' in line:
                line = line.replace('～',target)
                line_aa = re.split('丨|│|ㄧ', line)
            else:
                print(line)

            if CURRENT_L not in RESULTS_EX[CURRENT_W]:
                RESULTS_EX[CURRENT_W][CURRENT_L] = line_aa
            else:
                RESULTS_EX[CURRENT_W][CURRENT_L].extend(line_aa)

    RESULTS_AIM = set(RESULTS_AIM)

    return

def predict_sent(str_tst):
    global RESULTS_WS
    global RESULTS_FQ
    global RESULTS_STOP

    PADDING_ID = get_term_id(PADDING)

    str_tst = str_tst.strip()
    seg_list = pynlpir.segment(str_tst, pos_tagging=False)
    len_s = len(seg_list)
    for i in range(len_s):

        if seg_list[i] not in RESULTS_WS:
            print(seg_list[i], end=' ')
        else:
            # 准备周围词
            score_list = []
            if (i-1) >=0: env_pre = get_term_id(seg_list[i-1])
            else: env_pre = get_term_id(PADDING)
            if (i+1) <len_s: env_nex = get_term_id(seg_list[i+1])
            else: env_nex = get_term_id(PADDING)
            if env_pre == env_nex and env_pre == get_term_id(PADDING):
                print('%s/%s '%(seg_list[i], "_UNDEFINE_"), end='')
                continue

            if env_pre == PADDING_ID and seg_list[i+1] in RESULTS_STOP:
                pass
            elif env_nex == PADDING_ID and seg_list[i-1] in RESULTS_STOP:
                pass
            else:
                tmp_id = env_pre << TAG_SHIFT | env_nex 
                score_list.append(tmp_id)

            if (i-2) >= 0:
                env_pre =  get_term_id(seg_list[i-2])
                env_nex =  get_term_id(seg_list[i-1])
                if env_pre == PADDING_ID and seg_list[i-1] in RESULTS_STOP:
                    pass
                elif env_nex == PADDING_ID and seg_list[i-2] in RESULTS_STOP:
                    pass
                else:
                    tmp_id = env_pre << TAG_SHIFT | env_nex 
                    score_list.append(tmp_id)

            if (i+2) < len_s:
                env_pre =  get_term_id(seg_list[i+1])
                env_nex =  get_term_id(seg_list[i+2])
                if env_pre == PADDING_ID and seg_list[i+2] in RESULTS_STOP:
                    pass
                elif env_nex == PADDING_ID and seg_list[i+1] in RESULTS_STOP:
                    pass
                else:
                    tmp_id = env_pre << TAG_SHIFT | env_nex 
                    score_list.append(tmp_id)


            scores = {}
            for item in RESULTS_WS[seg_list[i]].keys():
                scores[item] = 0
                hit = 0
                for s_ls in score_list:
                    if s_ls not in RESULTS_FQ: continue
                    for w_s in RESULTS_WS[seg_list[i]][item]:
                        if w_s == seg_list[i]: continue
                        if get_term_id(w_s) in RESULTS_FQ[s_ls]:
                            scores[item] += RESULTS_FQ[s_ls][get_term_id(w_s)]
                            hit += 1
                            print(" %s-%s-[%s-%s]-%d " %(item, w_s, get_term_wd(s_ls >> TAG_SHIFT), get_term_wd(s_ls & TAG_MASK),RESULTS_FQ[s_ls][get_term_id(w_s)]))
                scores[item] = (scores[item] * hit) / len(RESULTS_WS[seg_list[i]][item])
            best_scores = sorted(scores.items(), key=lambda e:e[1], reverse=True)

            #print(best_scores)
            if best_scores[0][1] != 0:
                print('%s/%s '%(seg_list[i], best_scores[0][0]), end='')
            else:
                print('%s/%s '%(seg_list[i], "_UNDEFINE_"), end='')
    print()
    return

# Just handle the target wds
def predict_one_shot(str_tst, aimword):
    global RESULTS_WS
    global RESULTS_FQ
    global RESULTS_STOP

    PADDING_ID = get_term_id(PADDING)

    str_tst = str_tst.strip()
    seg_list = pynlpir.segment(str_tst, pos_tagging=False)
    len_s = len(seg_list)

    if aimword not in seg_list or len(seg_list) < 5:
        return None

    i = seg_list.index(aimword)

    # 准备周围词
    score_list = []
    if (i-1) >=0: env_pre = get_term_id(seg_list[i-1])
    else: env_pre = get_term_id(PADDING)
    if (i+1) <len_s: env_nex = get_term_id(seg_list[i+1])
    else: env_nex = get_term_id(PADDING)
    if env_pre == env_nex and env_pre == get_term_id(PADDING):
        return "_UNDEFINE_"

    if env_pre == PADDING_ID and seg_list[i+1] in RESULTS_STOP:
        pass
    elif env_nex == PADDING_ID and seg_list[i-1] in RESULTS_STOP:
        pass
    else:
        tmp_id = env_pre << TAG_SHIFT | env_nex 
        score_list.append(tmp_id)

    if (i-2) >= 0:
        env_pre =  get_term_id(seg_list[i-2])
        env_nex =  get_term_id(seg_list[i-1])
        if env_pre == PADDING_ID and seg_list[i-1] in RESULTS_STOP:
            pass
        elif env_nex == PADDING_ID and seg_list[i-2] in RESULTS_STOP:
            pass
        else:
            tmp_id = env_pre << TAG_SHIFT | env_nex 
            score_list.append(tmp_id)

    if (i+2) < len_s:
        env_pre =  get_term_id(seg_list[i+1])
        env_nex =  get_term_id(seg_list[i+2])
        if env_pre == PADDING_ID and seg_list[i+2] in RESULTS_STOP:
            pass
        elif env_nex == PADDING_ID and seg_list[i+1] in RESULTS_STOP:
            pass
        else:
            tmp_id = env_pre << TAG_SHIFT | env_nex 
            score_list.append(tmp_id)

    # Predict it!
    scores = {}
    for item in RESULTS_WS[seg_list[i]].keys():
        scores[item] = 0
        hit = 0
        for s_ls in score_list:
            if s_ls not in RESULTS_FQ: continue
            for w_s in RESULTS_WS[seg_list[i]][item]:
                if w_s == seg_list[i]: continue
                if get_term_id(w_s) in RESULTS_FQ[s_ls]:
                    scores[item] += (RESULTS_FQ[s_ls][get_term_id(w_s)] / sum(RESULTS_FQ[s_ls].values()))
                    hit += 1
        scores[item] = (scores[item] * hit) / len(RESULTS_WS[seg_list[i]][item])
        #scores[item] = ( hit) / len(RESULTS_WS[seg_list[i]][item])
    best_scores = sorted(scores.items(), key=lambda e:e[1], reverse=True)

    #print(best_scores)
    if best_scores[0][1] != 0:
        return best_scores[0][0]
    else:
        return "_UNDEFINE_"


def eval_alphago():
    global RESULTS_EX

    total = 0
    good  = 0

    for word in RESULTS_EX.keys(): #多义词
        for item in RESULTS_EX[word]: #意项
            for word_ls in RESULTS_EX[word][item]: #例句
                ret = predict_one_shot(word_ls, word)
                if ret == None or ret == "_UNDEFINE_":
                    continue

                if ret == item:
                    good += 1
                else:
                    print("%s %s %s"%(item, ret, word_ls))
                total += 1

    print("%d/%d" %(good, total))
    return

def dump_model(dump_file):
    global RESULTS_WS
    global RESULTS_AIM
    global RESULTS_FQ
    global RESULTS_EX

    print("DUMPING MODEL...")
    with open(dump_file,'wb', -1) as fp:
        dump_data = []
        dump_data.append(RESULTS_WS)
        dump_data.append(RESULTS_AIM)
        dump_data.append(RESULTS_FQ)
        dump_data.append(RESULTS_EX)
        pickle.dump(dump_data, fp, -1)
        del dump_data

    return

def load_model(dump_file):
    global RESULTS_WS
    global RESULTS_AIM
    global RESULTS_FQ
    global RESULTS_EX

    print("LOADING MODEL...")
    with open(dump_file,'rb', -1) as fp:
        dump_data = pickle.load(fp)
        RESULTS_WS = dump_data[0]
        RESULTS_AIM = dump_data[1]
        RESULTS_FQ = dump_data[2]
        RESULTS_EX = dump_data[3]
        del dump_data    

    return


if __name__ == '__main__':

    pynlpir.open()

    global DETECT_ERR
    DETECT_ERR = False

    global RESULTS_FQ
    global RESULTS_WS
    global RESULTS_AIM
    global RESULTS_STOP
    global RESULTS_EX

    load_dicts(DICT_FILE)

    if os.path.exists(MODEL_FILE):
        load_model(MODEL_FILE)
    else:

        process_tyccl() 
        print(len(RESULTS_AIM))
        print(RESULTS_WS.keys())

        #pynlpir.nlpir.Init(pynlpir.nlpir.PACKAGE_DIR, pynlpir.nlpir.UTF8_CODE, None)
        #for item in RESULTS_AIM:
        #    ret = pynlpir.nlpir.AddUserWord(item.encode())

        collect_env()
        print(len(RESULTS_FQ))   
        dump_model(MODEL_FILE)

    RESULTS_STOP = []
    with open('stopwords.txt', 'r') as fin:
        for line in fin:
            item = line.strip()
            if len(item.split()) > 1:
                print(item)
            if item not in RESULTS_AIM:
                RESULTS_STOP.append(item)
    RESULTS_STOP.append('了')
    RESULTS_STOP = set(RESULTS_STOP)

    predict_sent("我刚刚进了一批货，然后就这灯光看书籍，一点不觉得疲倦。哦！，我的钥匙卡了。。。")

    print("BEGIN EVAL ALPHAGO!")
    eval_alphago()

    if DETECT_ERR:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!ERROR DETECTED!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

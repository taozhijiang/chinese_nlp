#!/usr/bin/env python3

from hanzi_util import is_zhs, is_terminator
import jieba
from copy import deepcopy
import os
import pickle

STOP_FILE = 'stopwords.txt'
TYCC_FILE = 'TYCCL.txt'
TRAIN_COPS = 'train.txt'
#TRAIN_COPS = 'C08-财经.txt'
#TRAIN_COPS = 'all.txt'

#
# {   '单词': [{'意项1': W1, W2, ...}, {'意项2': W1, W2, ...}, ]   }
#

STOP_WORDS = []
SCAN_WORDS = []
TYCC_ITEMS = {}
TYCC_DAT = {}

PAD = 'P'

def split_to_sentnces(lst):
    ret = []
    len_t = len(lst)
    fro = 0
    for i in range(len_t):
        if is_terminator(lst[i]):
            ret.append(lst[fro:i])
            fro = i + 1
    return ret


def build_train_dat():
    global TYCC_DAT
    TYCC_DAT = {}

    line_num = 0
    with open(TRAIN_COPS) as fin:
        while True:
            try:
                each_line = fin.readline()
                if not each_line:
                    break_flag = True
                    print("处理完毕！")
                    break
                line_num += 1
                if not (line_num % 2000): print("C:%d" %(line_num))
                each_line = each_line.strip()

                seg_list = jieba.cut(each_line, cut_all=False)
                seg_list = split_to_sentnces(list(seg_list))
                for seg_ls in seg_list:
                    if not seg_ls: continue

                    words = []
                    for item in seg_ls:
                        if not is_zhs(item): continue
                        if item in STOP_WORDS: continue
                        words.append(item)

                    len_t = len(words)
                    if not len_t or len_t < 3: continue
                    for i in range(len_t):
                        if words[i] in SCAN_WORDS:
                            if words[i] not in TYCC_DAT:
                                TYCC_DAT[words[i]] = {}
                                if (i-2) >= 0:
                                    TYCC_DAT[words[i]][words[i-2]] = 1
                                    TYCC_DAT[words[i]][words[i-1]] = 2
                                elif (i-1) >= 0:
                                    TYCC_DAT[words[i]][words[i-1]] = 2
                                if (i+2) < len_t:
                                    TYCC_DAT[words[i]][words[i+2]] = 1
                                    TYCC_DAT[words[i]][words[i+1]] = 2
                                elif (i+1) < len_t:
                                    TYCC_DAT[words[i]][words[i+1]] = 2
                            else:
                                if (i-2) >= 0:
                                    if words[i-2] in TYCC_DAT[words[i]]:
                                        TYCC_DAT[words[i]][words[i-2]] += 1
                                    else:
                                        TYCC_DAT[words[i]][words[i-2]] = 1
                                    # for i - 1
                                    if words[i-1] in TYCC_DAT[words[i]]:
                                        TYCC_DAT[words[i]][words[i-1]] += 2
                                    else:
                                        TYCC_DAT[words[i]][words[i-1]] = 2
                                elif (i-1) >= 0:
                                    if words[i-1] in TYCC_DAT[words[i]]:
                                        TYCC_DAT[words[i]][words[i-1]] += 2
                                    else:
                                        TYCC_DAT[words[i]][words[i-1]] = 2

                                if (i+2) < len_t:
                                    if words[i+2] in TYCC_DAT[words[i]]:
                                        TYCC_DAT[words[i]][words[i+2]] += 1
                                    else:
                                        TYCC_DAT[words[i]][words[i+2]] = 1
                                    # for i + 1
                                    if words[i+1] in TYCC_DAT[words[i]]:
                                        TYCC_DAT[words[i]][words[i+1]] += 2
                                    else:
                                        TYCC_DAT[words[i]][words[i+1]] = 2
                                elif (i+1) < len_t:
                                    if words[i+1] in TYCC_DAT[words[i]]:
                                        TYCC_DAT[words[i]][words[i+1]] += 2
                                    else:
                                        TYCC_DAT[words[i]][words[i+1]] = 2

            except UnicodeDecodeError as e:
                print('Unicode Error! filename=%s, line_num=%d'%(TRAIN_COPS, line_num))
                pass


def build_model():
    global STOP_WORDS
    global SCAN_WORDS
    global TYCC_ITEMS

    STOP_WORDS = []
    with open(STOP_FILE, 'r') as fin:
        for line in fin:
            item = line.strip()
            if len(item.split()) > 1:
                print(item)
            if is_zhs(item):
                STOP_WORDS.append(item)

    STOP_WORDS = set(STOP_WORDS)
    print("STOP_WORDS:%d" %(len(STOP_WORDS)))

    STOP_WORDS = []

    SCAN_WORDS = []
    TYCC_ITEMS_PREP = {}
    with open(TYCC_FILE, 'r') as fin:
        for line in fin:
            items = line.strip().split()
            if len(items) < 3: continue
            if items[1][0] != '【' and items[1][-1] != '】': continue

            # 只注重同义词部分
            if items[1][-3:] != '.=】': continue

            # 目前只考虑动词
            if items[0] != 'v': continue

            words = []
            for item in items[2:]:
                if item not in STOP_WORDS:
                    words.append(item)
                #else:
                #    print('TRIM:%s' %(item))

            YX = items[1]
            if not len(words): continue

            for word in words:
                if word in TYCC_ITEMS_PREP:
                    item = TYCC_ITEMS_PREP[word]
                    item.append({YX:' '.join(words)}) 
                else:
                    TYCC_ITEMS_PREP[word] = [{YX:' '.join(words)}]

    print("TYCC_ITEMS ORIGINAL:%d" %(len(TYCC_ITEMS_PREP)))

    #无多意项剔除
    TYCC_ITEMS = deepcopy(TYCC_ITEMS_PREP)
    for word in TYCC_ITEMS_PREP:
        if len(TYCC_ITEMS_PREP[word]) == 1:
            TYCC_ITEMS.pop(word)
        else:
            for yxs in TYCC_ITEMS_PREP[word]:
                for(k, v) in yxs.items():
                    SCAN_WORDS.extend(v.split())
    del TYCC_ITEMS_PREP               
    print("TYCC_ITEMS AFTER:%d" %(len(TYCC_ITEMS)))  
    SCAN_WORDS = set(SCAN_WORDS) 
    print("SCAN_WORDS:%d" %(len(SCAN_WORDS)))


if __name__ == '__main__':

    dump_data = []
    if not os.path.exists("dump.dat"):
        build_model()
        build_train_dat()
        fp = open("dump.dat",'wb', -1)
        dump_data.append(STOP_WORDS)
        dump_data.append(SCAN_WORDS)
        dump_data.append(TYCC_ITEMS)
        dump_data.append(TYCC_DAT)
        pickle.dump(dump_data, fp, -1)
    else:
        fp = open("dump.dat",'rb')
        dump_data = pickle.load(fp)
        STOP_WORDS = dump_data[0]
        SCAN_WORDS = dump_data[1]
        TYCC_ITEMS = dump_data[2]
        TYCC_DAT = dump_data[3]
    del dump_data


    test_str = '亚行报告说，由于中国对部分出口商品主动采取了限制措施，加上国家减少了对出口商品间接补贴和本国劳动力成本上升等因素，预计中国今年货物出口增长将回落到２０％左右，继续上升的货物贸易顺差将部分地被服务贸易逆差所抵消。现行的财政政策对经济增长的刺激作用将减小。随着人民币新的汇率形成机制的启动，人民币大幅升值的投机性活动将明显减少。未来两年，尽管一些工业品价格可能继续下降，但水、电、气及石油产品价格将上涨，由此将拉动居民消费价格上升２至３个百分点。'

    print("ORIGINAL:" + test_str)
    seg_list = jieba.cut(test_str, cut_all=False)
    for item in seg_list:
        if item in TYCC_ITEMS:
            print(" [%s] "%(item), end='')
        else:
            print(item, end='')

    seg_list = split_to_sentnces(list(seg_list))
    for seg_ls in seg_list:
        if not seg_ls: continue

        words = []
        for item in seg_ls:
            if not is_zhs(item): continue
            if item in STOP_WORDS: continue
            words.append(item)

        len_t = len(words)
        if not len_t or len_t < 3: print("UN-SUPPORT!!!")
        for i in range(len_t):
            if words[i] in TYCC_ITEMS:  #有歧义词汇
                print(" [%s] "%(words[i]))
                p = '_P_'; n = '_P_';
                if (i-1) >= 0:
                    p = words[i-1]
                if (i+1) < len_t:
                    n = words[i+1]
                for yxs in TYCC_ITEMS[words[i]]:
                    for(k, v) in yxs.items():
                        print("\t"+k+"::")
                        chks = v.split()
                        for ck in chks:
                            if ck == words[i]: continue
                            if ck in TYCC_DAT:
                                if p in TYCC_DAT[ck]:
                                    print("%s-%s-%d"%(ck, p, TYCC_DAT[ck][p]))
                                if n in TYCC_DAT[ck]:
                                    print("%s-%s-%d"%(ck, n, TYCC_DAT[ck][n]))

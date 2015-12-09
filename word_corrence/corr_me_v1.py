#!/usr/bin/python3

import os
import jieba
import hanzi_util
import copy
import pickle
import math

CURRENT_VER = 1

CURR_DIR = os.getcwd()
DATA_DIR = os.getcwd() + "/../data_dir/tc-corpus-answer/" 
STOP_FILE  = CURR_DIR + "/../data_dir/stopwords.txt"

# 'word':[cnt: word-w1:N1 ,word-w2:N2 ]
train_data = {}
train_word_id = []
train_tags = []
stop_words = []


def term_to_id(term):
    if term not in train_word_id:
        train_word_id.append(term)
    voca_id = train_word_id.index(term)
    return voca_id

def build_train_data():
    global train_data
    global train_tags
    global stop_words
    global train_word_id
    train_data = {}
    train_tags = []
    stop_words = []
    train_word_id = []
    
    with open(STOP_FILE, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line[0] == '#': continue
            stop_words.append(line)
    print("STOP WORD SIZE:%d\n" %(len(stop_words)))
        
    for parent,dirname,filenames in os.walk(DATA_DIR):
        for filename in filenames:
            tag_name = filename[:-4]
            print("正在处理：%s"%(tag_name))
            train_tags.append(tag_name)
            line_num = 0
            with open(DATA_DIR+'/'+filename,'r') as fin:
                for line in fin:
                    line_num += 1
                    if not line_num % 1000 : print('LINE:%d'%(line_num))
                    line = line.strip()
                    line_t = jieba.cut(line, cut_all=False)
                    objs = []
                    for item in line_t:
                        if item not in stop_words and hanzi_util.is_zhs(item):
                            item_id = term_to_id(item)
                            if item_id not in objs: 
                                objs.append(item_id)
                            if item_id in train_data.keys():
                                train_data[item_id]['COUNT'] += 1
                            else:
                                #print("ADDING ITEM:%s" %(item));
                                train_data[item_id] = {}
                                train_data[item_id]['COUNT'] = 1
                            if tag_name not in train_data[item_id].keys():
                                train_data[item_id][tag_name] = {}

                    #公现指数计算
                    #我们只计算一个方向的
                    if len(objs) < 2: continue
                    #print(objs)
                    for index_i in range(len(objs) - 1):
                        for index_j in range(index_i + 1, len(objs)):
                            #print('%d-%d-%d'%(len(objs),index_i, index_j))
                            item_i = objs[index_i]
                            item_j = objs[index_j]
                            item_t = item_i<<32 | item_j
                            if item_t in train_data[item_i][tag_name].keys():
                                train_data[item_i][tag_name][item_t] += 1
                            else:   #反向的是否存在
                                train_data[item_i][tag_name][item_t] = 1
    
    return

def calc_vector(str):
    count_all = {}
    sub_train = []
    if not str or not len(str):
        return None
    line = str.strip()
    line_t = jieba.cut(line, cut_all=False)
    objs = []
    for item in line_t:
        if item not in stop_words and hanzi_util.is_zhs(item):
            if item not in train_word_id:
               continue
            item_id = term_to_id(item)
            if item_id not in objs:
                objs.append(item_id)
    if len(objs) < 2: return None
    
    for index_i in range(len(objs) - 1):
        for index_j in range(index_i + 1, len(objs)):
            item_i = objs[index_i]
            item_j = objs[index_j]
            item_t = item_i<<32 | item_j
            sub_train.append(item_t)

    for item_tag in train_tags:
        count_all[item_tag] = {}
        for item_w in sub_train:
            item_1 = item_w >> 32
            item_2 = item_w  & 0xFFFFFFFF
            item_a = item_1 << 32 | item_2
            item_b = item_2 << 32 | item_1
            count_s = train_data[item_1]['COUNT'] + train_data[item_2]['COUNT']
            count = 0
            if item_1 in train_data.keys() and \
            item_tag in train_data[item_1].keys() and \
            item_a in train_data[item_1][item_tag].keys():
                count +=  train_data[item_1][item_tag][item_a]
            if item_2 in train_data.keys() and \
            item_tag in train_data[item_2].keys() and \
            item_b in train_data[item_2][item_tag].keys():
                count +=  train_data[item_2][item_tag][item_b]

            #这里将对数值取反，绝对值越小，概率越大
            count_all[item_tag][item_w] = -math.log(count / count_s + 0.0000000001)
    return count_all        
        
def test_sub(str):
    test_str = str
    if not str or not len(str):  return
    print("测试文本：")
    print(test_str)
    data = calc_vector(test_str)
    print("测试结果：")
    if data:
        predict = {}
        for (tag, val) in data.items():
            predict[tag] = sum(val.values())/ len(val)
        #概率排序
        sorted_list = sorted(predict.items(), key=lambda e:e[1], reverse=False)
        for tag in sorted_list:
            print("%20s: %f" %(tag))
    
if __name__ == '__main__':
    
    dump_file = "./dump_data.dat_v%d"%(CURRENT_VER)
    if not os.path.exists(dump_file):
        print("BUILDING DATA....")
        build_train_data()
        with open(dump_file,'wb', -1) as fp:
            dump_data = []
            dump_data.append(train_data)
            dump_data.append(train_tags)
            dump_data.append(stop_words)
            dump_data.append(train_word_id)
            pickle.dump(dump_data, fp, -1)
            del dump_data
    else:
        print("LOADING DATA....")
        with open(dump_file,'rb', -1) as fp:
            dump_data = pickle.load(fp)
            train_data = dump_data[0]
            train_tags = dump_data[1]
            stop_words = dump_data[2]
            train_word_id = dump_data[3]
            del dump_data
        print("字典长度：%d" %(len(train_word_id)))

    train_file = "./train_data.dat_v%d"%(CURRENT_VER)
    if not os.path.exists(train_file):        
        print('DUMP训练结果')
        train_len = len(train_word_id)
        index = 0
        with open(train_file, 'w') as fin:
            for (item, val) in train_data.items():
                index += 1
                if not index % 1000: print('%d/%d'%(index, train_len))
                #开头索引词
                words = train_word_id[item & 0xFFFFFFFF] + ':\n'
                #fin.write(words)
                for tag in train_tags:
                    fin.write(tag+':\n')
                    if tag in train_data[item].keys():
                        for (item_sub, item_cnt) in train_data[item][tag].items():
                            item_words = train_word_id[item_sub>>32] + '~' + train_word_id[item_sub & 0xFFFFFFFF]
                            fin.write("%s:%d  "%(item_words, item_cnt))


        
    test_str = '面对令人鼓舞的石油开采前景和目前可观的石油收入，赤几政府认为'
    test_sub(test_str)
    test_str = '在山东省滨州市一家培训学校进行的艺考模拟测试上,艺考生在进行'
    test_sub(test_str)
    test_str = '足球是巴西人文化生活的主流。对巴西人来说，足球是运动，但更是文化。'
    test_sub(test_str)
    test_str = '叙利亚政府军遭到不同武装团体的反抗。其中最为积极的为“伊斯兰国”和“胜利阵线“的武装分子。'
    test_sub(test_str)
    test_str = '中国国家主席习近平偕夫人彭丽媛乘专机抵达津巴布韦首都哈拉雷,开始对津巴布韦进行国事访问。'
    test_sub(test_str)
    test_str = '苹果iPhone 5s今日在商家“苹果手机专卖店处热促,最新报价为1699元。该机配件为单电、数据线、耳机、充电器等标配。'
    test_sub(test_str)

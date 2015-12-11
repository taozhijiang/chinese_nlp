#!/usr/bin/python3

#项目更新
# train[hi-lo][tags] 
#感觉一个词的词，对区分作用有限

CURRENT_VER = 2

import os
import jieba
import hanzi_util
import copy
import pickle
import math

import socket
import sys

CURR_DIR = os.getcwd()
DATA_DIR = os.getcwd() + "/../data_dir/tc-corpus-all/" 
STOP_FILE  = CURR_DIR + "/../data_dir/stopwords.txt"
WHITE_FILE  = CURR_DIR + "/../data_dir/whitewords.txt"

#出现的单个词的词表ID
train_word_id = []
#single记录单个词在各个TAG下的出现频率
# word_id:[ tag1:count, tag2:count ...]
train_data_single = {}
#训练的总体数据 hi-word_id: [ low-word_id : [tag1:count tag2:count ...] ]
#降低内存使用， low-word<<12 2^12 == 4K 个分类，足够了
tag_shift = 12
tag_mask  = 0xFFF
train_data = {}
#标签列表，从1开始索引
train_tags = ['NULL']  
stop_words = []
white_words = []

debug_s_words = []

def term_to_id(term):
    if term not in train_word_id:
        train_word_id.append(term)
    voca_id = train_word_id.index(term)
    return voca_id

def build_train_data():
    global train_word_id
    global train_data_single
    global train_data
    global train_tags
    global stop_words
    train_word_id = []
    train_data_single = {}
    train_data = {}
    train_tags = ['NULL']
    stop_words = []
    
    with open(STOP_FILE, 'r') as fin:
        for line in fin:
            line = line.strip()
            if not line or line[0] == '#': continue
            stop_words.append(line)
    print("STOP WORD SIZE:%d\n" %(len(stop_words)))
     
    with open(WHITE_FILE, 'r') as fin:
        for line in fin:
            line = line.strip()
            if not line or line[0] == '#': continue
            white_words.append(line)
    print("WHITE WORD SIZE:%d\n" %(len(white_words)))

    for parent,dirname,filenames in os.walk(DATA_DIR):
        for filename in filenames:
            tag_name = filename[:-4]
            print("正在处理：%s"%(tag_name))
            train_tags.append(tag_name)
            tag_id = train_tags.index(tag_name)
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
                            if len(item) == 1 and item not in white_words:
                                if item not in debug_s_words: debug_s_words.append(item)
                                continue
                            item_id = term_to_id(item)
                            if item_id not in objs: 
                                objs.append(item_id)

                            # train_data_single
                            if not item_id in train_data_single:
                                train_data_single[item_id] = {}
                            if not tag_id in train_data_single[item_id]:
                                train_data_single[item_id][tag_id] = 1
                            else:
                                train_data_single[item_id][tag_id] += 1

                    # train_data
                    #公现指数计算
                    #我们只计算一个方向的，且排列按照 index 低-高 排列
                    if len(objs) < 2: continue
                    #print(objs)
                    for index_i in range(len(objs) - 1):
                        for index_j in range(index_i + 1, len(objs)):
                            #print('%d-%d-%d'%(len(objs),index_i, index_j))
                            if objs[index_i] < objs[index_j]:
                                item_i = objs[index_i]
                                item_j = objs[index_j]
                            else:
                                item_i = objs[index_j]
                                item_j = objs[index_i]
                            item_j_tag = (item_j << tag_shift ) | tag_id
                            if item_i in train_data:
                                if item_j_tag in train_data[item_i]:
                                    train_data[item_i][item_j_tag] += 1
                                else:
                                    train_data[item_i][item_j_tag] = 1
                            else:
                                train_data[item_i] = {}
                                train_data[item_i][item_j_tag] = 1
                                
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
            if item not in train_word_id:  # 单字词已经被踢掉了
               continue
            item_id = term_to_id(item)
            if item_id not in objs:
                objs.append(item_id)
    if len(objs) < 2: return None
    
    for index_i in range(len(objs) - 1):
        for index_j in range(index_i + 1, len(objs)):
            if objs[index_i] < objs[index_j]:
                item_i = objs[index_i]
                item_j = objs[index_j]
            else:
                item_i = objs[index_j]
                item_j = objs[index_i]
            item_t = item_i<<32 | item_j
            sub_train.append(item_t)

    for item_tag in train_tags[1:]:
        tag_id = train_tags.index(item_tag)
        count_all[item_tag] = {}
        for item_w in sub_train:
            item_1 = item_w >> 32
            item_2 = item_w  & 0xFFFFFFFF
            item_2_tag = (item_2 << tag_shift) | tag_id
            count_s = 0
            #count_s = sum(train_data_single[item_1].values()) + sum(train_data_single[item_2].values())
            if tag_id in train_data_single[item_1]:
                count_s += train_data_single[item_1][tag_id]
            if tag_id in train_data_single[item_2]:
                count_s += train_data_single[item_2][tag_id]
            count = 0
            if item_1 in train_data and \
            item_2_tag in train_data[item_1]:
                count +=  train_data[item_1][item_2_tag]

            #这里将对数值取反，绝对值越小，概率越大
            if count_s == 0 or count == 0:
                count_all[item_tag][item_w] = -math.log(0.0000000001)
            else:
                count_all[item_tag][item_w] = -math.log(count / count_s + 0.0000000001)
    return count_all        
        
def test_sub(str, put):
    test_str = str
    sorted_list = None
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
        sorted_list = sorted_list[:5]
        if put:
            for tag in sorted_list:
                print("%20s: %f" %(tag))
    return sorted_list

    
if __name__ == '__main__':
    
    dump_file = "./dump_data.dat_v%d"%(CURRENT_VER)
    if not os.path.exists(dump_file):
        print("BUILDING DATA....")
        build_train_data()
        print(debug_s_words)
        del debug_s_words
        with open(dump_file,'wb', -1) as fp:
            dump_data = []
            dump_data.append(train_word_id)
            dump_data.append(train_data_single)
            dump_data.append(train_data)
            dump_data.append(train_tags)
            dump_data.append(stop_words)
            dump_data.append(white_words)
            pickle.dump(dump_data, fp, -1)
            del dump_data
    else:
        print("LOADING DATA....")
        with open(dump_file,'rb', -1) as fp:
            dump_data = pickle.load(fp)
            train_word_id = dump_data[0]
            train_data_single = dump_data[1]
            train_data = dump_data[2]
            train_tags = dump_data[3]
            stop_words = dump_data[4]
            white_words = dump_data[5]
            del dump_data
        print("字典长度：%d" %(len(train_word_id)))

    train_file = "./train_data.dat_v%d"%(CURRENT_VER)
    if not os.path.exists(train_file):        
        print('DUMP训练结果')
        train_len = len(train_word_id)
        index = 0
        with open(train_file, 'w') as fout:
            for (item_1, item_1val) in train_data.items():
                index += 1
                if not index % 1000: print('%d/%d'%(index, train_len))
                #开头索引词
                word_1 = train_word_id[item_1]
                fout.write(word_1+':\n')
                if item_1val:
                    for (item_2, item_2val) in item_1val.items():
                        word_2 = train_word_id[item_2 >> tag_shift]
                        tag_id = item_2 & tag_mask
                        fout.write('\t'+word_2+'['+train_tags[tag_id]+']:'+str(item_2val)+'\n')

    
    if False:    
        test_str = '面对令人鼓舞的石油开采前景和目前可观的石油收入，赤几政府认为'
        test_sub(test_str, 1)
        test_str = '在山东省滨州市一家培训学校进行的艺考模拟测试上,艺考生在进行'
        test_sub(test_str, 1)
        test_str = '足球是巴西人文化生活的主流。对巴西人来说，足球是运动，但更是文化。'
        test_sub(test_str, 1)
        test_str = '叙利亚政府军遭到不同武装团体的反抗。其中最为积极的为“伊斯兰国”和“胜利阵线“的武装分子。'
        test_sub(test_str, 1)
        test_str = '中国国家主席习近平偕夫人彭丽媛乘专机抵达津巴布韦首都哈拉雷,开始对津巴布韦进行国事访问。'
        test_sub(test_str, 1)
        test_str = '苹果iPhone 5s今日在商家“苹果手机专卖店处热促,最新报价为1699元。该机配件为单电、数据线、耳机、充电器等标配。'
        test_sub(test_str, 1)
    
    #每次启动需要加载的数据比较多，这里设置成服务端，接受客户端的请求
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = ''   #local nic
    port = 34774

    sock.bind((host, port))
    sock.listen(10)

    print("服务端OK，侦听请求：")
    while True:
        conn, addr = sock.accept()
        data = conn.recv(4096).decode().strip()
        if data:
            ret = test_sub(data, 0)
            if ret:
                conn.sendall(repr(ret).encode())
            else:
                conn.sendall('计算为空...'.encode())
        else:
            print('请求为空...')
        conn.close()

    sock.close()

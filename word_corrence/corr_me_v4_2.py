#!/usr/bin/python3

#项目更新
# train[hi-lo][tags] 
#感觉一个词的词，对区分作用有限

CURRENT_VER = 4

import os
import jieba
import hanzi_util
import copy
import pickle
import math

import socket
import sys

from random import shuffle

import nltk
# 统计词频
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures

# 各种机器学习算法
import sklearn
from nltk.classify.maxent import MaxentClassifier
from sklearn.metrics import accuracy_score

CURR_DIR = os.getcwd()
DATA_DIR = os.getcwd() + "/../data_dir/ClassFile_4000_4/" 
STOP_FILE  = CURR_DIR + "/../data_dir/stopwords.txt"
WHITE_FILE  = CURR_DIR + "/../data_dir/whitewords.txt"

#出现的单个词的词表ID
train_word_id = []
train_info = {}
#标签列表，从1开始索引
train_tags = ['NULL']  
stop_words = []
white_words = []
best_words = []
sorted_word_scores = {}

def term_to_id(term):
    global train_word_id
    if term not in train_word_id:
        train_word_id.append(term)
    voca_id = train_word_id.index(term)
    return voca_id

def find_best_words(num):
    global sorted_word_scores
    if not sorted_word_scores or num <= 0:
        print("INVALID ARGUMENT...")
        return None
    if num < len(sorted_word_scores):
        tmp_word_scores = sorted_word_scores[:num]
    else:
        tmp_word_scores = sorted_word_scores
    best_words  = set([w for w, s in tmp_word_scores])
    return best_words

def best_word_features(words, b_words):
    if not b_words: return None
    return dict([(word, True) for word in words if word in b_words])

def build_train_data():
    global train_word_id
    global train_data_single
    global train_info
    global train_tags
    global stop_words
    train_word_id = []
    train_data_single = {}
    train_info = {}
    train_tags = ['NULL']
    stop_words = []

    word_fd = FreqDist() #可统计所有词的词频
    cond_word_fd = ConditionalFreqDist() #可统计积极文本中的词频和消极文本中的词频
    
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
            if filename[-6:] != '_p.txt': continue
            tag_name = filename[:-6]
            print("正在处理：%s"%(tag_name))
            train_tags.append(tag_name)
            tag_id = train_tags.index(tag_name)
            train_info[tag_id] = []
            line_num = 0
            with open(DATA_DIR+'/'+filename,'r') as fin:
                for line in fin:
                    line_num += 1
                    if not line_num % 500 : print('LINE:%d'%(line_num))
                    line = line.strip()
                    objs = []
                    for item in line.split():
                        if len(item) == 1 and item not in white_words:
                            continue
                        item_id = term_to_id(item)
                        if item_id not in objs: 
                            word_fd[item_id] += 1
                            cond_word_fd[tag_id][item_id] += 1
                            objs.append(item_id)
                    train_info[tag_id].append(objs)

    print('Randomize>>>')
    cond_word_sum = {}
    for tag in train_tags[1:]:
        tag_id = train_tags.index(tag)
        shuffle(train_info[tag_id])  
        cond_word_sum[tag_id] = cond_word_fd[tag_id].N()
        print("SUM:%s->%d"%(tag, cond_word_sum[tag_id]))
    total_w_count = word_fd.N() 
    print("TOTAL:%d"%(total_w_count))

    global sorted_word_scores
    sorted_word_scores = {}
    word_scores = {}
               
    word_scores_sub = {} 
    print("CALC CHI-SQUARE...")
    for word, freq in word_fd.items():
        word_scores[word] = 0
        for tag in train_tags[1:]:
            tag_id = train_tags.index(tag)
            word_scores[word] += \
            BigramAssocMeasures.chi_sq(cond_word_fd[tag_id][word], (freq, cond_word_sum[tag_id]), total_w_count)
    sorted_word_scores = sorted(word_scores.items(), key=lambda e:e[1], reverse=True)
    
    del cond_word_sum
    del word_fd
    del cond_word_fd

    return

def build_classifier(trainSet):
    if not trainSet:
        return None
    classifier = MaxentClassifier.train(trainSet, algorithm='gis')
    return classifier
        
def final_score(classifier, test, tag_test):
    #print("LABEL:"+repr(sorted(classifier.labels())))
    pred = classifier.classify_many(test)
    return accuracy_score(tag_test, pred)

def final_prob(classifier, data_str):
    count_all = {}
    if not data_str or not len(data_str):
        return None
    line = data_str.strip()
    line_t = jieba.cut(line, cut_all=False)
    objs = []
    for item in line_t:
        if item not in stop_words and hanzi_util.is_zhs(item):
            if item not in train_word_id:  # 单字词已经被踢掉了
               continue
            item_id = term_to_id(item)
            if item_id not in objs:
                objs.append(item_id)

    test_feature = best_word_features(objs, best_words)
    if not test_feature: 
        print('特征为空...')
        return None
    prob = classifier.prob_classify(test_feature)
    return prob   

def enum_params():     
    ###########################################
    #
    # FEATURE & ARGS TEST
    ###########################################
    feature_n = [2000, 4000, 6000, 8000, 10000, 13000, 16000, 20000, 25000, 30000, 40000, 50000]

    print(repr(train_info))
    #对原始语料进行训练和测试分割
    len_all = len(train_info[1])
    tra_len = int(len_all *0.9)
    tst_len = int(len_all *0.1)

    for f_n in feature_n:
        best_words = find_best_words(f_n)

        train_feature = []
        test_feature =[]
        for tag in train_tags[1:]:
            tag_id = train_tags.index(tag)
            for item in train_info[tag_id][:tra_len]:
                train_feature.append((best_word_features(item, best_words),tag_id))
            for item in train_info[tag_id][tra_len+1:]:
                test_feature.append((best_word_features(item, best_words),tag_id))

        classifier = build_classifier( train_feature)
        test, tag_test = zip(*test_feature)
        res = final_score(classifier, test, tag_test)
        print('%8d:%f\t'%(f_n, res))
    os._exit(0)

if __name__ == '__main__':

    BEST_N = 20000
    
    dump_file_data = "./dump_data.dat_v%d"%(CURRENT_VER)
    dump_file_class = "./dump_class.dat_v%d"%(CURRENT_VER)

    if not os.path.exists(dump_file_class):
        if not os.path.exists(dump_file_data):
            print("BUILDING DATA....")
            build_train_data()

            print("STORING DATA....")
            with open(dump_file_data,'wb', -1) as fp:
                dump_data = []
                dump_data.append(train_word_id)
                dump_data.append(train_tags)
                dump_data.append(stop_words)
                dump_data.append(white_words)
                dump_data.append(train_info)
                dump_data.append(sorted_word_scores)
                pickle.dump(dump_data, fp, -1)
                del dump_data
        else:
            print("LOADING DATA....")
            with open(dump_file_data,'rb', -1) as fp:
                dump_data = pickle.load(fp)
                train_word_id = dump_data[0]
                train_tags = dump_data[1]
                stop_words = dump_data[2]
                white_words = dump_data[3]
                train_info = dump_data[4]
                sorted_word_scores = dump_data[5]
                del dump_data

        #############################################
        #############################################
        #############################################
        enum_params()

        print("BUILDING CLASSIFIER....")
        best_words = find_best_words(BEST_N)

        #对原始语料进行训练和测试分割
        print("BUILDING TRAIN AND TEST FEATURES...")
        len_all = len(train_info[1])
        tra_len = int(len_all *0.9)
        tst_len = int(len_all *0.1)
        print("tra_len:%d, tst_len:%d" %(tra_len, tst_len))

        train_feature = []
        test_feature =[]
        for tag in train_tags[1:]:
            tag_id = train_tags.index(tag)
            print("DOING... %s"%(tag))
            for item in train_info[tag_id][:tra_len]:
                train_feature.append((best_word_features(item, best_words),tag_id))
            for item in train_info[tag_id][tra_len+1:]:
                test_feature.append((best_word_features(item, best_words),tag_id))

        print("TRAINING CLASSIFIER...")
        classifier = build_classifier(train_feature)
        test, tag_test = zip(*test_feature)
        res = final_score(classifier, test, tag_test)
        print('MultinomialNB:%f'%(res))
        
        print("TRAINED LABEL:"+repr(sorted(classifier.labels())))

        print("STORING CLASSIFIER....")
        with open(dump_file_class,'wb', -1) as fp:
            dump_data = []
            dump_data.append(train_word_id)
            dump_data.append(train_tags)
            dump_data.append(stop_words)
            dump_data.append(white_words)
            dump_data.append(best_words)
            dump_data.append(classifier)
            pickle.dump(dump_data, fp, -1)
            del dump_data
        print("字典长度：%d" %(len(best_words)))

    else:
        print("LOADING CLASSIFIER....")
        with open(dump_file_class,'rb', -1) as fp:
            dump_data = pickle.load(fp)
            train_word_id = dump_data[0]
            train_tags = dump_data[1]
            stop_words = dump_data[2]
            white_words = dump_data[3]
            best_words = dump_data[4]
            classifier = dump_data[5]
            del dump_data
        print("字典长度：%d" %(len(best_words)))


    #每次启动需要加载的数据比较多，这里设置成服务端，接受客户端的请求
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host = ''   #local nic
    port = 34772

    sock.bind((host, port))
    sock.listen(10)

    print("服务端OK，侦听请求：")
    while True:
        conn, addr = sock.accept()
        data_str = conn.recv(4096).decode().strip()
        if data_str:
            ret = final_prob(classifier, data_str)
            if ret:
                ret_str = ''
                for it in train_tags[1:]:
                    ret_str += 'TAG:%s, PROB:%f\n' %(it, ret.prob(train_tags.index(it)))
                conn.sendall(ret_str.encode())
            else:
                conn.sendall('计算为空...'.encode())
        else:
            print('请求为空...')
        conn.close()

    sock.close()

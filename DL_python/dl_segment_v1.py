#!/usr/bin/python3

import os
import pickle
import nltk
import numpy as np
import sys
sys.setrecursionlimit(1000000) #设置为一百万，否则dump数据的时候显示递归层次过多！


# 切分数据集
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils
from keras.models import Sequential,Graph
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Reshape, Flatten , Dense, Dropout, Activation

tagindex = {'B':0, 'E':1, 'M':2, 'S':3 }
indextag = {0:'B', 1:'E', 2:'M', 3:'S'}
# dict的查找速度比List会快一些
word2index = {}
index2word = {}
wordvector = {}

TRAIN_FILE = "../data_dir/icwb2-data/training/msr_pku_lite.utf8"
#TRAIN_FILE = "../data_dir/icwb2-data/training/pku_training.utf8"
#TRAIN_FILE = "../data_dir/icwb2-data/training/msr_pku_training.utf8"

def wordindex(word):
    if not word: return None
    if word not in word2index:
        len_t = len(word2index)
        word2index[word] = len_t
        index2word[len_t] = word
    return word2index[word]

# 给定的句子，生成对应的训练向量，包含首位的PADING c c c W c c c 
# 输入是list状态的单个句子(分词的或者未分词的)
def sent2num(sentence, context = 7):
    word_num = []
    for item in sentence:
        for w in item:
            # 文本中的字如果在词典中则转为数字，如果不在则设置为'U
            if w in word2index:
                word_num.append(word2index[w])
            else:
                word_num.append(word2index['U'])
    # 首尾padding
    num = len(word_num)
    pad = int((context-1)*0.5) #3 by default
    for i in range(pad):
        word_num.insert(0,word2index['P'] )
        word_num.append(word2index['P'] )
    train_x = []
    for i in range(num):
        train_x.append(word_num[i:i+context])
    return train_x    

# 给定的句子，生成对应的TAG S B M E   
def sent2tag(sentence):
    train_tg = []
    for item in sentence:
        if len(item) == 1:
            train_tg.append(tagindex['S'])
            continue
        train_tg.append(tagindex['B'])
        for w in item[1:len(item)-1]:
            train_tg.append(tagindex['M']) 
        train_tg.append(tagindex['E']) 
    return train_tg   
    
# 根据输入得到标注推断
def predict_num(input_txt, input_num, model):
    str_ret = '';
    input_num = np.array(input_num)
    predict_prob = model.predict_proba(input_num, verbose=False)
    predict_lable = model.predict_classes(input_num, verbose=False)
    for i , lable in enumerate(predict_lable[:-1]):
        # 如果是首字 ，不可为E, M
        if i == 0:
            predict_prob[i, tagindex['E']] = 0
            predict_prob[i, tagindex['M']] = 0      
        # 前字为B，后字不可为B,S
        if lable == tagindex['B']:
            predict_prob[i+1,tagindex['B']] = 0
            predict_prob[i+1,tagindex['S']] = 0
        # 前字为E，后字不可为M,E
        if lable == tagindex['E']:
            predict_prob[i+1,tagindex['M']] = 0
            predict_prob[i+1,tagindex['E']] = 0
        # 前字为M，后字不可为B,S
        if lable == tagindex['M']:
            predict_prob[i+1,tagindex['B']] = 0
            predict_prob[i+1,tagindex['S']] = 0
        # 前字为S，后字不可为M,E
        if lable == tagindex['S']:
            predict_prob[i+1,tagindex['M']] = 0
            predict_prob[i+1,tagindex['E']] = 0
        predict_lable[i+1] = predict_prob[i+1].argmax()
        
    #predict_lable_new = [indextag[x]  for x in predict_lable]
    #result =  [w+'/' +l  for w, l in zip(input_txt,predict_lable_new)]
    for i in range(len(input_txt)):
        str_ret += input_txt[i]
        if predict_lable[i] == tagindex['S'] or predict_lable[i] == tagindex['E']:
            str_ret += ' '
        
    return str_ret
    
def build_dl_model():   
    input_file_str = []
    
    line_num = 0
    with open(TRAIN_FILE) as fin:
        try:
            for each_line in fin:
                if not each_line:
                    continue
                line_num += 1
                if not (line_num % 2000): print("C:%d" %(line_num))
                line_items = each_line.split()
                for item in line_items:
                    for w in item:
                        wordindex(w) # USE THE SIDE EFFECT, 单个字的vector
                        
                input_file_str.append(line_items)
        except UnicodeDecodeError as e:
            print('Unicode Error! filename=%s, line_num=%d'%(TRAIN_FILE, line_num))

    print("训练长度：%d" %(len(input_file_str)))

    len_t = len(word2index)
    word2index['U'] = len_t
    index2word[len_t] = 'U'
    word2index['P'] = len_t+1
    index2word[len_t+1] = 'P'

    len_input_file_str = len(input_file_str)

    train_vector = []
    train_tag    = []
    # 再次遍历训练数据
    for i_input in range(len_input_file_str):
        #for item_j in input_file_str[i_input]:
        train_x = sent2num(input_file_str[i_input])
        train_g = sent2tag(input_file_str[i_input])
        train_vector.extend(train_x)    
        train_tag.extend(train_g)


    print("SIZE VECTOR:%d TAG:%d" %(len(train_vector), len(train_tag)))

    train_vector = np.array(train_vector)
    train_X, test_X, train_y, test_y = train_test_split(train_vector, train_tag , train_size=0.90, random_state=1)
    print(len(train_X), 'train sequences')
    print(len(test_X), 'test sequences')


    batch_size = 128
    maxfeatures = len(index2word)
    word_dim = 100
    maxlen = 7
    hidden_units = 100
    nb_classes = 4

    Y_train = np_utils.to_categorical(train_y, nb_classes)
    Y_test  = np_utils.to_categorical(test_y, nb_classes)

    print('Stacking LSTM...')
    model = Sequential()
    model.add(Embedding(maxfeatures, word_dim, input_length=maxlen))
    model.add(LSTM(output_dim=hidden_units, return_sequences =True))
    model.add(LSTM(output_dim=hidden_units, return_sequences =False))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print("Train...")
    result = model.fit(train_X, Y_train, batch_size=batch_size, 
                       nb_epoch=4, validation_data = (test_X,Y_test), show_accuracy=True)
    score = model.evaluate(test_X, Y_test, batch_size=batch_size)
    print("Test Score:%d" %(score))
    
    return model


if __name__ == "__main__":
    
    DUMP_FILE = "dump.dat_v1"    
    
    if os.path.exists(DUMP_FILE):
        print("LOADING DL...")

        dump_data = []
        with open(DUMP_FILE,'rb', -1) as fin:
            dump_data = pickle.load(fin)
            word2index = dump_data[0]
            index2word = dump_data[1]
            dl_model = dump_data[2]   
        print("DONE!")
        
    else:
        print("BUILDING DL...")

        dl_model = build_dl_model()
        
        dump_data = []
        with open(DUMP_FILE,'wb', -1) as fout:
            dump_data.append(word2index)
            dump_data.append(index2word)
            dump_data.append(dl_model)
            pickle.dump(dump_data, fout, -1);   
        print("DONE!")
        
    temp_txt = '国家食药监总局发布通知称，酮康唑口服制剂因存在严重肝毒性不良反应，即日起停止生产销售使用。'
    temp_txt = list(temp_txt)
    temp_num = sent2num(temp_txt)
    ret = predict_num(temp_txt, temp_num, dl_model)  
    print(ret)
    
    temp_txt = "首先是个民族问题，民族的感情问题"
    temp_txt = list(temp_txt)
    temp_num = sent2num(temp_txt)
    ret = predict_num(temp_txt, temp_num, dl_model)     
    print(ret)

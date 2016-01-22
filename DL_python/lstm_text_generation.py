#!/usr/bin/python3


import os
import sys
sys.setrecursionlimit(1000000) #设置为一百万，否则dump数据的时候显示递归层次过多！
import pickle

import numpy as np
import random
np.random.seed(25535) 

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from hanzi_util import is_zh

# skip-gram and CBOW models
from gensim.models import word2vec

# store and load
from keras.models import model_from_json  
from keras.callbacks import EarlyStopping  

word2index = {}
index2word = {}
wordvector = []
input_str = []
train_vector = []
train_nextw  = []

TRAIN_SENT_LEN = 8  #8个词一句话

CORPUS_FILE = "../data_dir/jd_comm_mixed/lite.txt"

def wordindex(word):
    global word2index
    global index2word
    if not word: return None
    if word not in word2index:
        len_t = len(word2index)
        word2index[word] = len_t
        index2word[len_t] = word
    return word2index[word]

def getwordindex(word):
    global word2index
    if word in word2index: return word2index[word]
    else: return word2index['U']

def build_word2vec(filename): 
    global word2index
    global index2word
    global wordvector
    global input_str
    word2index = {}
    index2word = {} 
    wordvector = []
    
    word2vec_str = []
    
    line_num = 0
    with open(filename) as fin:
        while True:
            try:
                each_line = fin.readline()
                if not each_line:
                    break_flag = True
                    print("处理完毕！")
                    break
                line_num += 1
                if not (line_num % 500): print("C:%d" %(line_num))
                line_items = each_line.split()
                wvecs = []
                for item in line_items:
                    for w in item:
                        wordindex(w) # USE THE SIDE EFFECT, 单个字的vector
                        wvecs.append(w)
                word2vec_str.append(wvecs)
            except UnicodeDecodeError as e:
                print('Unicode Error! filename=%s, line_num=%d'%(filename, line_num))
                pass
            
    print("词库长度：%d"%(len(word2index)))
    print("训练长度：%d"%(len(word2vec_str)))

    w2v_model = word2vec.Word2Vec(sentences=word2vec_str, size=100, window=4, \
                              min_count=2 , workers=4, sorted_vocab=1, iter=10)
    print("word2vec DONE.")

    # 保存向量
    len_t = len(word2index)
    for i in range(len_t):
        if index2word[i] in w2v_model:
            wordvector.append(w2v_model[index2word[i]])
        else:
            #print("RANDOM VECTOR FOR: %s" %(index2word[i]))
            wordvector.append(np.random.randn(100,))
            
    # 定义'U'为未登陆新字, 'P'为padding用途
    word2index['U'] = len_t
    index2word[len_t] = 'U'
    word2index['P'] = len_t+1
    index2word[len_t+1] = 'P'
    wordvector.append(np.random.randn(100,))
    wordvector.append(np.zeros(100,))
    
    input_str = word2vec_str
    print("WORD2VEC_LEN:%d, STR_LEN:%d" %(len(wordvector), len(input_str)))
    
    return
    
    

def build_sent_batch():
    global input_str
    global train_vector
    global train_nextw
    train_vector = []
    train_nextw  = []

    for item_sent in input_str:
        if not len(item_sent): continue
        if len(item_sent) <= TRAIN_SENT_LEN:
            this_sent = []            
            for i in range(0, TRAIN_SENT_LEN + 1 - len(item_sent)):  # reserve 1 word next
                this_sent.append(word2index['P'])
            for i in range(0, len(item_sent)-1):
                this_sent.append(getwordindex(item_sent[i]))
            train_vector.append(this_sent)
            train_nextw.append(getwordindex(item_sent[-1]))
        else:
            item_len = len(item_sent)
            for i in range(0, item_len - TRAIN_SENT_LEN): # reserve 1 word next
                this_sent = []
                for j in range(i, i + TRAIN_SENT_LEN):
                    this_sent.append(getwordindex(item_sent[j]))
                train_vector.append(this_sent)
                train_nextw.append(getwordindex(item_sent[i + TRAIN_SENT_LEN]))
                    
                    
    print("TRAIN_VECTOR: %d" %(len(train_vector)))
    print("TRAIN_NEXTW: %d" %(len(train_nextw)))
    return

def build_lstm_mode():
    global train_vector
    global train_nextw
    global wordvector

    batch_size = 128
    maxfeatures = len(wordvector)
    word_dim = 100
    maxlen = TRAIN_SENT_LEN
    hidden_units = 100
    

    train_vector = np.array(train_vector)
    train_nextw = np.array(train_nextw)     
    
    print('Stacking LSTM...')
    model = Sequential()
    model.add(Embedding(input_dim = maxfeatures, output_dim = word_dim, input_length=maxlen, weights=[np.array(wordvector)]))
    model.add(LSTM(output_dim=hidden_units, return_sequences =True))
    model.add(LSTM(output_dim=hidden_units, return_sequences =False))
    model.add(Dropout(0.5))
    model.add(Dense(maxfeatures))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')    
    
    print("Train...")
    Y_train = np_utils.to_categorical(train_nextw, maxfeatures)
    result = model.fit(train_vector, Y_train, batch_size=batch_size, nb_epoch=20) 
    
    return model
    
def gen_sentence(model, sent_num, min_len):
    global wordvector
    
    print("About to generate sentence...")
    for i in range(sent_num):
        start_seq = []
        for j in range(0, TRAIN_SENT_LEN):
            start_seq.append(random.randint(0, len(word2index)))
        
        result = ""
        while True:            
            x = np_utils.to_categorical(start_seq, len(word2index))
            preds = model.predict(x)[0]            
            next_index = np.argmax(preds)
            next_char = index2word[next_index]            
            if next_index == word2index['P'] or next_index == word2index['U']: #or not is_zh(next_char):
                start_seq[-1] = random.randint(0, len(word2index))
                continue
            start_seq = start_seq[1:]
            start_seq.append(next_index)
            result += next_char
            if len(result) >= min_len:
                print("%d:%s" %(i, result))
                result = ""
                break
    

if __name__ == '__main__':

    DUMP_FILE           = "./dump_dir/lstm_text_wordindex.dat_v2" 
    model_json_fname    = "./dump_dir/lstm_text_model.json_v2"
    model_weights_fname = "./dump_dir/lstm_text_model.weights_v2"   
    
    if os.path.exists(DUMP_FILE) and os.path.exists(model_json_fname) and os.path.exists(model_weights_fname):
        print("LOADING DL...")

        dump_data = []
        with open(DUMP_FILE,'rb', -1) as fin:
            dump_data = pickle.load(fin)
            word2index = dump_data[0]
            index2word = dump_data[1]  
        dl_model = model_from_json(open(model_json_fname).read()) 
        dl_model.load_weights(model_weights_fname) 

        print("DONE!")
        
    else:
        print("BUILDING DL...")
        
        build_word2vec(CORPUS_FILE)
        build_sent_batch()
        dl_model = build_lstm_mode()
        
        dump_data = []
        with open(DUMP_FILE,'wb', -1) as fout:
            dump_data.append(word2index)
            dump_data.append(index2word)
            pickle.dump(dump_data, fout, -1);   
        json_string = dl_model.to_json()
        open(model_json_fname,'w').write(json_string)  
        dl_model.save_weights(model_weights_fname)   
        print("DONE!")     
    
    gen_sentence(dl_model, 5, 8)
    print('DONE!')
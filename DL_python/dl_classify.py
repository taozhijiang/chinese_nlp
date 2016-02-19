#!/usr/bin/python3

import os
import pickle
import nltk
import numpy as np
np.random.seed(25535)  
import sys
import jieba
import hanzi_util
from random import shuffle

import nltk
# 统计词频
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures

# 切分数据集
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils
from keras.models import Sequential,Graph
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Reshape, Flatten , Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop

# skip-gram and CBOW models
from gensim.models import word2vec

# store and load
from keras.models import model_from_json  
from keras.callbacks import EarlyStopping  


# 有用词表
word2index = {}
index2word = {}
train_vector = []
train_vtag = []
wordvector = []
#标签列表，从0开始索引
train_tags = []  

DATA_DIR = os.getcwd() + "/../data_dir/ClassFile_L/"

def wordindex(word):
    if not word: return None
    if word not in t_word2index:
        len_t = len(t_word2index)
        t_word2index[word] = len_t
        t_index2word[len_t] = word
    return t_word2index[word]

def build_train_data(best_n):
    global word2index
    global index2word
    global train_vector
    global train_vtag
    global train_tags
    global wordvector
    word2index = {}
    index2word = {}
    train_tags = []
    wordvector = []

    t_word2index = {}
    t_index2word = {}
    t_train_info = {}
    word2vec_str = []

    print("BUILD TRAIN DATA...")

    word_fd = FreqDist() #可统计所有词的词频
    cond_word_fd = ConditionalFreqDist() #可统计积极文本中的词频和消极文本中的词频

    for parent,dirname,filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if filename[-4:] != '.txt': continue
            tag_name = filename[:-4]
            print("正在处理：%s"%(tag_name))
            train_tags.append(tag_name)
            tag_id = train_tags.index(tag_name)
            t_train_info[tag_id] = []
            line_num = 0
            with open(DATA_DIR+'/'+filename,'r') as fin:
                while True:
                    try:
                        line = fin.readline()
                    except UnicodeDecodeError as e:
                        print('Unicode Error! filename=%s, line_num=%d'%(filename, line_num))
                        continue 
                    if not line:
                        print('文件已处理完! filename=%s, line_num=%d'%(filename, line_num))
                        break

                    line_num += 1
                    if not line_num % 500 : print('LINE:%d'%(line_num))
                    line = line.strip()
                    line_t = jieba.cut(line, cut_all=False)
                    objs = []
                    objs_str = []
                    for item in line_t:
                        if hanzi_util.is_zh(item[0]):
                            if item not in t_word2index:
                                item_id = len(t_word2index)
                                t_word2index[item] = item_id
                                t_index2word[item_id] = item
                            else:
                                item_id = t_word2index[item]
                            if item_id not in objs: 
                                word_fd[item_id] += 1
                                cond_word_fd[tag_id][item_id] += 1
                                objs.append(item_id)
                            objs_str.append(item)
                    if objs:
                        t_train_info[tag_id].append(objs)
                    if objs_str:
                        word2vec_str.append(objs_str)

    w2v_model = word2vec.Word2Vec(sentences=word2vec_str, size=100, window=4, \
                              min_count=3 , workers=4, sorted_vocab=1, iter=10)
    del word2vec_str
    print("word2vec DONE.")

    print('Randomize>>>')
    cond_word_sum = {}
    for tag in train_tags:
        tag_id = train_tags.index(tag)
        shuffle(t_train_info[tag_id])  
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
        for tag in train_tags:
            tag_id = train_tags.index(tag)
            word_scores[word] += \
            BigramAssocMeasures.chi_sq(cond_word_fd[tag_id][word], (freq, cond_word_sum[tag_id]), total_w_count)
    sorted_word_scores = sorted(word_scores.items(), key=lambda e:e[1], reverse=True)
    
    del cond_word_sum
    del word_fd
    del cond_word_fd

    if best_n < len(sorted_word_scores):
        sorted_word_scores = sorted_word_scores[:best_n]
    else:
        best_n = len(sorted_word_scores)

    # real word2index index2world
    for index in range(best_n):
        word2index[t_index2word[sorted_word_scores[index][0]]] = index
        index2word[index] = t_index2word[sorted_word_scores[index][0]]
        
    for i in range(len(word2index)):
        if index2word[i] in w2v_model:
            wordvector.append(w2v_model[index2word[i]])
        else:
            print("RANDOM VECTOR FOR: %s" %(index2word[i]))
            wordvector.append(np.random.randn(100,))
     
    # 'U'
    word2index['U'] = best_n
    index2word[best_n] = 'U'        
    wordvector.append(np.random.randn(100,))

    train_vector = []
    train_vtag = []

    # train_info
    for tag_id  in t_train_info:
        for l_index in range(len(t_train_info[tag_id])):  
            objs = []
            for item in t_train_info[tag_id][l_index]:
                if t_index2word[item] in word2index:
                    objs.append(word2index[t_index2word[item]])
                else:
                    objs.append(word2index['U'])
            objs = list(set(objs))
            if objs:
                train_vector.append(objs)
                train_vtag.append(tag_id)
            
            
    del t_index2word
    del t_word2index
    del sorted_word_scores
    del t_train_info

    print("build_train_data finished!")
    return

def to_categorical_s(y, nb_classes):
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        items = y[i]
        for item in items:
            Y[i, item] = 1.
    return Y

    
def build_dl_model():
    global train_vector
    global train_vtag
    global train_tags
    global wordvector

    best_n = len(word2index) # including 'U'

    train_vector = to_categorical_s(train_vector, best_n)
    train_X, test_X, train_y, test_y = train_test_split(train_vector, train_vtag , train_size=0.90, random_state=1)
    print(len(train_X), 'train sequences')
    print(len(test_X), 'test sequences')

    maxfeatures = best_n
    word_dim = 100
    batch_size = 128
    hidden_units = 512
    nb_classes = len(train_tags)

    Y_train = np_utils.to_categorical(train_y, nb_classes)
    Y_test  = np_utils.to_categorical(test_y, nb_classes)

    print('Stacking LSTM...')
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(maxfeatures,)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_units))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    print("Train...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)  
    result = model.fit(train_X, Y_train, batch_size=batch_size, 
                       nb_epoch=10, validation_data = (test_X,Y_test), show_accuracy=True,
                       callbacks=[early_stopping])
    
    return model
    

def predict_class(model, data_str):

    best_n = len(word2index)
    count_all = {}
    if not data_str or not len(data_str):
        return None
    line = data_str.strip()
    line_t = jieba.cut(line, cut_all=False)
    objs = []
    for item in line_t:
        if hanzi_util.is_zh(item[0]):
            if item not in word2index:  # 单字词已经被踢掉了
                objs.append(word2index['U'])
            else:
                objs.append(word2index[item])
        else:
            objs.append(word2index['U'])
    objs = list(set(objs))
    objs_vector = np.zeros((1, best_n))
    for item in objs:
        objs_vector[0, item] = 1.
    predict_prob = model.predict_proba(objs_vector, verbose=False)

    return predict_prob.argmax()

if __name__ == "__main__":

    BEST_N = 5000
    DUMP_FILE           = "./dump_dir/dl_classify_wordindex.dat_v2" 
    model_json_fname    = "./dump_dir/dl_classify_model.json_v2"
    model_weights_fname = "./dump_dir/dl_classify_model.weights_v2"   
    
    if os.path.exists(DUMP_FILE) and os.path.exists(model_json_fname) and os.path.exists(model_weights_fname):
        print("LOADING DL...")

        dump_data = []
        with open(DUMP_FILE,'rb', -1) as fin:
            dump_data = pickle.load(fin)
            word2index = dump_data[0]
            index2word = dump_data[1]
            train_tags = dump_data[2]
        dl_model = model_from_json(open(model_json_fname).read()) 
        dl_model.load_weights(model_weights_fname) 

        print("DONE!")
        
    else:
        print("BUILDING RAW DL...")

        build_train_data(BEST_N)
        dl_model = build_dl_model()
        
        dump_data = []
        with open(DUMP_FILE,'wb', -1) as fout:
            dump_data.append(word2index)
            dump_data.append(index2word)
            dump_data.append(train_tags)
            pickle.dump(dump_data, fout, -1);   
        json_string = dl_model.to_json()
        open(model_json_fname,'w').write(json_string)  
        dl_model.save_weights(model_weights_fname)   
        print("DONE!") 

    test_list = ['中东和平的建设者、中东发展的推动者、中东工业化的助推者、中东稳定的支持者、中东民心交融的合作伙伴——习近平主席在演讲中为中国-中东关系发展指明的方向，切合地区实际情况，照顾地区国家关切，为摆在国际社会面前的“中东之问”给出了中国的答案。', \
            '2014年6月，习近平在中阿合作论坛北京部长级会议上提出，中阿共建“一带一路”，构建以能源合作为主轴，以基础设施建设、贸易和投资便利化为两翼，以核能、航天卫星、新能源三大高新领域为新的突破口的“1+2+3”合作格局。', \
            '在此次落马的16人里面，级别最高的是连城县委原书记江国河。履历显示，江国河1963年出生，龙岩市永定县高头乡人。被调查时，他已在福建省能源集团有限责任公司董事、纪委书记的位子上干了两年。', \
            '机智堂是新浪手机推出的新栏目，风趣幽默是我们的基调，直白简单地普及手机技术知识是我们的目的。我们谈手机，也谈手机圈的有趣事，每月定期更新，搞机爱好者们千万不能错过。']    
    
    print(train_tags)
    for temp_txt in test_list:
        ret = predict_class(dl_model, temp_txt)  
        print("[%s]: %s" %(train_tags[ret], temp_txt))

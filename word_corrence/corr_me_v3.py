#!/usr/bin/python3

#项目更新
# train[hi-lo][tags] 
#感觉一个词的词，对区分作用有限
#分开多线程跑
# 由于PYTHON语言的原因，测试这里分成多个线程根本没有用到多核计算机的
# 的优势，其实还是在单个核心上模拟的多线程操作，执行效率没有提升，反而
# 因为上下文切换的原因降低执行效率
# 后面用multiprocess，虽然可以用多线程，但是进程间交换数据的开销变得非常
# 的大，所以：除非IO阻塞复用，否则Python不适合多线程开发

CURRENT_VER = 3

import os
import jieba
import hanzi_util
import copy
import pickle
import math
import threading
import socket
import sys
import queue

CURR_DIR = os.getcwd()
DATA_DIR = os.getcwd() + "/../data_dir/tc-corpus-answer/" 
STOP_FILE  = CURR_DIR + "/../data_dir/stopwords.txt"
WHITE_FILE  = CURR_DIR + "/../data_dir/whitewords.txt"
TMP_PATH = CURR_DIR + "/tmp/"

#出现的单个词的词表ID
train_word_id = []
#tag1: data,  tag2:data ...
train_data = {}
#标签列表，从1开始索引
train_tags = ['NULL']  
stop_words = []
white_words = []

debug_s_words = []

q = queue.Queue()
gdb_lock = threading.RLock()

def term_to_id(term):
    if term not in train_word_id:
        #写保护
        with gdb_lock:
            if term not in train_word_id:
                train_word_id.append(term)
    voca_id = train_word_id.index(term)
    return voca_id

class TrainThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    
    def run(self):
        while True:
            try:
                tag_name = q.get(timeout =  5)
            except queue.Empty as e:
                print("Task Queue is empty, return!")
                return
            print("Thread-%d正在处理:%s，还剩:%d"%(self.threadID, tag_name, q.qsize()))
            #if os.path.exists(TMP_PATH+tag_name+'.dat'):
            #    print("DAT %s already exits, skip it!"%(tag_name))
            #    q.task_done()
            #    continue
            line_num = 0
            fast_prep = 1
            sub_train_data = {}
            if os.path.exists(DATA_DIR+'/'+tag_name+'_p.txt'):
                open_file = DATA_DIR+'/'+tag_name+'_p.txt'
                fast_prep = 1
            else:
                open_file = DATA_DIR+'/'+tag_name+'.txt'
                fast_prep = 0
            with open(open_file,'r') as fin:
                while True:
                    try:
                        line = fin.readline()
                    except UnicodeDecodeError as e:
                        print('Unicode Error! thread=%d, tag=%s, line_num=%d'%(self.threadID, tag_name, line_num))
                        continue 
                    if not line:
                        print('文件已处理完! thread=%d, tag=%s, line_num=%d'%(self.threadID, tag_name, line_num))
                        break
                    line_num += 1
                    if not line_num % 1000 : print('Thread-%d,LINE:%d'%(self.threadID, line_num))
                    objs = []
                    if not fast_prep:
                        line = line.strip()
                        line_t = jieba.cut(line, cut_all=False)
                        for item in line_t:
                            if item not in stop_words and hanzi_util.is_zhs(item):
                                if len(item) == 1 and item not in white_words:
                                    continue
                                item_id = term_to_id(item)
                                if item_id not in objs: 
                                    objs.append(item_id)
                    else:
                        for item in line.split():
                            if len(item) == 1 and item not in white_words:
                                continue
                            item_id = term_to_id(item)
                            if item_id not in objs: 
                                objs.append(item_id)
                            #objs = [ term_to_id(t_id) for t_id in line.split()]
                    if len(objs) < 2: continue
                    for index_i in range(len(objs) - 1):
                        for index_j in range(index_i + 1, len(objs)):
                            #print('%d-%d-%d'%(len(objs),index_i, index_j))
                            if objs[index_i] < objs[index_j]:
                                item_i = objs[index_i]
                                item_j = objs[index_j]
                            else:
                                item_i = objs[index_j]
                                item_j = objs[index_i]
                            if item_i in sub_train_data:
                                if item_j in sub_train_data[item_i]:
                                    sub_train_data[item_i][item_j] += 1
                                else:
                                    sub_train_data[item_i][item_j] = 1
                            else:
                                sub_train_data[item_i] = {}
                                sub_train_data[item_i][item_j] = 1

            # 数据量太大，将出现频次小于等于1的词剔除掉
            print("精简数据...")
            iter_obj = copy.deepcopy(sub_train_data)
            for item_1 in iter_obj.keys():
                if not iter_obj[item_1]: continue
                for item_2 in iter_obj[item_1].keys():
                    if iter_obj[item_1][item_2] <= 1:
                        #print("DEBUG1:%d - %s/%s" %(iter_obj[item_1][item_2] ,train_word_id[item_1], train_word_id[item_2]))
                        del sub_train_data[item_1][item_2]
                if not iter_obj[item_1]:
                    print("DEBUG2:%s" %(train_word_id[item_1]))
                    del sub_train_data[item_1]
            del iter_obj

            print("保存数据...")
            # sub_train_data
            dump_file = TMP_PATH+tag_name+'.dat'
            with open(dump_file,'wb', -1) as fp:
                pickle.dump(sub_train_data, fp, -1)
            del sub_train_data
            print("Thread-%d处理[%s]结束!"%(self.threadID, tag_name))               
            q.task_done()
    
def build_train_data():
    global train_word_id
    global train_tags
    global stop_words
    global white_words
    global train_data
    train_word_id = []
    train_tags = ['NULL']
    stop_words = []
    white_words = []
    train_data = {}
    
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

    #统计总的标签数目
    for parent,dirname,filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if filename[-4:] != '.txt': continue
            if filename[-6:] == '_p.txt': continue
            tag_name = filename[:-4]
            train_tags.append(tag_name)
            q.put(tag_name)

    print(train_tags)
    if not os.path.exists(TMP_PATH): os.mkdir(TMP_PATH)
    #分发给各个消费线程去处理
    threads = []
    for i in range(10,12):
        t = TrainThread(i)
        #t.setDaemon(True)
        t.start()
        threads.append(t)

    # Wait until all the tag_train has been processed
    q.join()
    # 将dump的数据集合起来
    train_data = {}
    print('COLLECTING DATA...')
    for tag_name in train_tags[1:]:
        print('正在处理:'+tag_name)
        dump_file = TMP_PATH+tag_name+'.dat'
        with open(dump_file,'rb', -1) as fp:
            train_data[tag_name] = pickle.load(fp)
    print('DONE!')
    return

def calc_vector(str):
    count_all = {}
    sub_train = []
    pair_debug = {}
    if not str or not len(str):
        return (None,None)
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
    if len(objs) < 2: return (None,None)
    
    #产生搭配组合
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

    #统计总出现次数
    count_all = {}
    for item_tag in train_tags[1:]:
        tag_val = train_data[item_tag]
        count_all[item_tag] = {}
        for item_w in sub_train:
            count_all[item_tag][item_w] = 0
            item_1 = item_w >> 32
            item_2 = item_w  & 0xFFFFFFFF
            if item_1 in tag_val and item_2 in tag_val[item_1]:
                count_all[item_tag][item_w] += tag_val[item_1][item_2]
    count_s = {}
    for item_w in sub_train:
        count_s[item_w] = 0
        for item_tag in train_tags[1:]:
            count_s[item_w] += count_all[item_tag][item_w] 
    
    print(count_s)
    print(count_all)
    #计算概率 
    count_ret = {}
    for item_tag in train_tags[1:]:
        count_ret[item_tag] = {}
        pair_debug[item_tag] = {}
        for item_w in sub_train:
            item_1 = item_w >> 32
            item_2 = item_w  & 0xFFFFFFFF

            #这里将对数值取反，绝对值越小，概率越大
            count = count_all[item_tag][item_w] 
            if count == 0 or count_s[item_w] == 0:
                count_ret[item_tag][item_w] = - math.log(0.0000000001)
            else:
                count_ret[item_tag][item_w] = - count*math.log(count / count_s[item_w] + 0.0000000001)
                pair_debug[item_tag][train_word_id[item_1]+'~'+train_word_id[item_2]] = '%d/%d' %(count, count_s[item_w])
    return (count_ret, pair_debug)        
        
def test_sub(str, put):
    test_str = str
    sorted_list = None
    data_d = None
    if not str or not len(str):  return
    print("测试文本：")
    print(test_str)
    (data, data_d) = calc_vector(test_str)
    print("测试结果：")
    if data:
        predict = {}
        for (tag, val) in data.items():
            predict[tag] = sum(val.values()) / len(val)
        #概率排序
        sorted_list = sorted(predict.items(), key=lambda e:e[1], reverse=False)
        sorted_list = sorted_list[:5]
        if put:
            for tag in sorted_list:
                print("%20s: %f" %(tag))
            print(data_d)

    return (sorted_list, data_d)
    
if __name__ == '__main__':

    dump_file = "./dump_data.dat_v%d"%(CURRENT_VER)
    if not os.path.exists(dump_file):
        print("BUILDING DATA....")
        build_train_data()
        print(debug_s_words)
        del debug_s_words
        print("DUMPING DATA...")
        with open(dump_file,'wb', -1) as fp:
            dump_data = []
            dump_data.append(train_word_id)
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
            train_data = dump_data[1]
            train_tags = dump_data[2]
            stop_words = dump_data[3]
            white_words = dump_data[4]
            del dump_data
        print("字典长度：%d" %(len(train_word_id)))

    train_file = "./train_data.dat_v%d"%(CURRENT_VER)
    if not os.path.exists(train_file):        
        print('DUMP训练结果')
        with open(train_file, 'w') as fout:
            for tag_name in train_tags[1:]:
                tag_val = train_data[tag_name]
                fout.write('TAG:'+tag_name+':\n')
                if tag_val:
                    for (item_1, item_1val) in tag_val.items():
                        fout.write('\t'+train_word_id[item_1]+':\n')
                        if item_1val:
                            for (item_2, item_2val) in item_1val.items():
                                fout.write('\t\t'+train_word_id[item_1]+'-'+train_word_id[item_2]+':'+str(item_2val)+'\n')
                                
                                
    #测试
    test_str = '面对令人鼓舞的石油开采前景和目前可观的石油收入，赤几政府认为'
    test_sub(test_str, 1)
    
    #每次启动需要加载的数据比较多，这里设置成服务端，接受客户端的请求
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = ''   #local nic
    port = 34772

    sock.bind((host, port))
    sock.listen(10)

    print("服务端OK，侦听请求：")
    while True:
        conn, addr = sock.accept()
        data = conn.recv(4096).decode().strip()
        if data:
            print('测试字串:'+data)
            (ret, ret_d) = test_sub(data, 0)
            if ret:
                for key in ret:
                    print(repr(key)+":")
                    print(ret_d[key[0]])
                conn.sendall(repr(ret).encode())
            else:
                conn.sendall('计算为空...'.encode())
        else:
            print('请求为空...')
        conn.close()

    sock.close()


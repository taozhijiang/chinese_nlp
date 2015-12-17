#!/usr/bin/python3
import os

import nltk
# 统计词频
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures

from random import shuffle

import pickle
import socket

# 各种机器学习算法
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
#from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import jieba
import hanzi_util

STOP_FILE = os.getcwd() + "/../data_dir/stopwords.txt"
DATA_DIR = os.getcwd() + "/../data_dir/jd_comm_mixed/" 

# 1-8000 训练集合
# 8001-10000 测试集合
pos_file = DATA_DIR+'good_lite.txt'
neg_file = DATA_DIR+'bad_lite.txt'
#pos_file = DATA_DIR+'good_p.txt'
#neg_file = DATA_DIR+'bad_p.txt'
pos_info = []
neg_info = []

word_scores = {}

stop_words = []
best_words = []

#出现的单个词的词表ID
train_word_id = []
def term_to_id(term):
    if term not in train_word_id:
    		train_word_id.append(term)
    voca_id = train_word_id.index(term)
    return voca_id

def find_best_words(num):
	if not word_scores or num <= 0:
		return None
	#根据卡方统计量，对结果进行排序，选出较好具有较强区分能力的词
	best_scores = sorted(word_scores.items(), key=lambda e:e[1], reverse=True)
	if num < len(best_scores):
		best_scores = best_scores[:num]
	best_words  = set([w for w, s in best_scores])
	return best_words

def best_word_features(words, b_words):
	if not b_words: return None
	return dict([(word, True) for word in words if word in b_words])

def cal_word_count():
	global train_word_id
	global pos_info
	global neg_info
	pos_info = []
	neg_info = []
	train_word_id = []

	word_fd = FreqDist() #可统计所有词的词频
	cond_word_fd = ConditionalFreqDist() #可统计积极文本中的词频和消极文本中的词频

	print('Loading POS>>>')
	line_num = 0
	with open(pos_file, 'r') as fin:
		for line in fin:
			line_num += 1
			if not line_num % 10000: print('LINE:%d'%(line_num))
			items = line.split()
			tmp_col = []
			for item in items:
				item_id = term_to_id(item)
				word_fd[item_id] += 1
				cond_word_fd['pos'][item_id] += 1
				tmp_col.append(item_id)
			pos_info.append(tmp_col)

	print('Loading NEG>>>')
	line_num = 0
	with open(neg_file, 'r') as fin:
		for line in fin:
			line_num += 1
			if not line_num % 10000: print('LINE:%d'%(line_num))
			items = line.split()
			tmp_col = []
			for item in items:
				item_id = term_to_id(item)
				word_fd[item_id] += 1
				cond_word_fd['neg'][item_id] += 1
				tmp_col.append(item_id)
			neg_info.append(tmp_col)

	print('Randomize>>>')
	shuffle(pos_info)
	shuffle(neg_info)	

	pos_w_count = cond_word_fd['pos'].N()
	neg_w_count = cond_word_fd['neg'].N()
	total_w_count = pos_w_count + neg_w_count
	#print('pos_w_count=%d, neg_w_count=%d, total_w_count=%d'%(pos_w_count, neg_w_count, total_w_count))
	#print('word_fd_count=%d'%(word_fd.N()))

	#计算卡方统计量
	global word_scores
	word_scores = {}

	print("CALC CHI-SQUARE...")
	for word, freq in word_fd.items():
		pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_w_count), total_w_count) #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
		neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_w_count), total_w_count) #同理
		word_scores[word] = pos_score + neg_score #一个词的信息量等于积极卡方统计量加上消极卡方统计量

	del word_fd
	del cond_word_fd

	return

def build_classifier(o_classifier, trainSet):
	if not o_classifier or not trainSet:
		return None
	classifier = SklearnClassifier(o_classifier)
	classifier.train(trainSet)
	return classifier
		

def final_score(classifier, test, tag_test):
	print("LABEL:"+repr(sorted(classifier.labels())))
	pred = classifier.classify_many(test)
	return accuracy_score(tag_test, pred)

def final_prob(classifier, str_test):
	if not classifier or not str_test:
		return None
	str_test = str_test.strip()
	line_t = jieba.cut(str_test, cut_all=False)
	objs = []
	for item in line_t:
		if item not in stop_words and hanzi_util.is_zhs(item) and item in train_word_id:
			item_id = term_to_id(item)
			if item_id not in objs: 
				objs.append(item_id)
	if not objs: return None
	feat = best_word_features(objs, best_words)
	if not feat: return None
	prob = classifier.prob_classify(feat)
	return prob


# PARAMETER:
BEST_N = 2000
#MultinomialNB, BernoulliNB
#LogisticRegression
CLASSIFIER = LogisticRegression()

if __name__ == '__main__':

	dump_file = './dump_data.dat'
	if not os.path.exists(dump_file):

		print("BUILDING THE CLASSIFIER>>>")

		stop_words = []
		with open(STOP_FILE, 'r') as fin:
				for line in fin:
					line = line.strip()
					if not line or line[0] == '#': continue
					stop_words.append(line)

		print("STOP WORD SIZE:%d\n" %(len(stop_words)))
		
		cal_word_count()
		print("FINDING BEST WORDS....")
		best_words = find_best_words(BEST_N)

		#对原始语料进行训练和测试分割
		print("BUILDING POS AND NEG FEATURES...")
		len_all = len(pos_info)
		tra_len = int(len_all *0.8)
		tst_len = int(len_all *0.2)
		pos_feature = []
		neg_feature = []
		print("POS...")
		for item in pos_info[:tra_len]:
			pos_feature.append((best_word_features(item, best_words),'pos'))
		print("NEG...")			
		for item in neg_info[:tra_len]:
			neg_feature.append((best_word_features(item, best_words),'neg'))

		# Free Memory
		del pos_info
		del neg_info

		train_set = pos_feature[:tra_len]+neg_feature[:tra_len]
		test_set = pos_feature[-tst_len+1:]+neg_feature[-tst_len+1:]

		print("BUILDING CLASSIFIER...")
		classifier = build_classifier(CLASSIFIER, train_set)

		print("DUMPING RESULTS...")
		with open(dump_file,'wb', -1) as fp:
			dump_data = []
			dump_data.append(train_word_id)
			dump_data.append(stop_words)
			dump_data.append(classifier)
			dump_data.append(test_set)
			dump_data.append(best_words)
			pickle.dump(dump_data, fp, -1)
			del dump_data
	else:

		print("LOADING THE CLASSIFIER>>>")

		with open(dump_file,'rb', -1) as fp:
			dump_data = pickle.load(fp)
			train_word_id = dump_data[0]
			stop_words = dump_data[1]
			classifier = dump_data[2]
			test_set = dump_data[3]
			best_words = dump_data[4]
			del dump_data


	test, tag_test = zip(*test_set)
	res = final_score(classifier, test, tag_test)
	print('BernoulliNB:%f'%(res))

	test_str = '售后服务真差真真真差！'
	print(test_str)
	res = final_prob(classifier, test_str)
	if res:
		print('pos:%f, neg:%f' %(res.prob('pos'), res.prob('neg')))

	test_str = '这个手机很好,很好,很好,很好，很好用'
	print(test_str)
	res = final_prob(classifier, test_str)
	if res:
		print('pos:%f, neg:%f' %(res.prob('pos'), res.prob('neg')))		


	#每次启动需要加载的数据比较多，这里设置成服务端，接受客户端的请求
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	host = ''   #local nic
	port = 34770

	sock.bind((host, port))
	sock.listen(10)

	print("服务端OK，侦听请求：")
	while True:
	    conn, addr = sock.accept()
	    data_str = conn.recv(4096).decode().strip()
	    if data_str:
	        ret = final_prob(classifier, data_str)
	        if ret:
	            ret_str = 'pos:%f, neg:%f' %(ret.prob('pos'), ret.prob('neg'))
	            conn.sendall(ret_str.encode())
	        else:
	            conn.sendall('计算为空...'.encode())
	    else:
	        print('请求为空...')
	    conn.close()

	sock.close()

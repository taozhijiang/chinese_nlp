#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import numpy as np
import hanzi_util
import pickle

import jieba
from gensim import corpora, models, similarities

DATA_DIR   = os.getcwd() + "/../data_dir/" 
STOP_FILE  = DATA_DIR + "stopwords.txt"
#TRAIN_FILE = DATA_DIR + "icwb2-data/training/msr_training.utf8"
TRAIN_FILE = DATA_DIR + "jd.txt"

stop_words = []
train_set  = []

dump_data = []

if not os.path.exists("./dump.dat"):
	with open(STOP_FILE, 'r') as fin:
		stop_words = []
		for line in fin:
			line = line.strip()
			stop_words.append(line)

	with open(TRAIN_FILE, 'r') as fin:
		train_set = []
		for line in fin:
			#line = line.strip().split()
			line = line.strip()
			line = jieba.cut(line, cut_all=False)
			obj = []
			for item in line:
				if item not in stop_words and hanzi_util.is_zhs(item):
					obj.append(item)
			train_set.append(obj)
			
	#stop
	fp = open("./dump.dat",'wb', -1)
	dump_data = []
	dump_data.append(stop_words)
	dump_data.append(train_set)
	pickle.dump(dump_data, fp, -1)
	
else:
	fp = open("./dump.dat",'rb')
	dump_data = pickle.load(fp)
	stop_words = dump_data[0]
	train_set = dump_data[1]

print(len(stop_words))
print(len(train_set))
	
		
dic 	= corpora.Dictionary(train_set)
corpus 	= [dic.doc2bow(text) for text in train_set]
# Transforming vectors
tfidf 	= models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
# id2word is a mapping from word ids (integers) to words (strings)
lda 	= models.LdaModel(corpus_tfidf, id2word = dic, num_topics = 20)
print(lda)
#corpus_lda = lda[corpus_tfidf]

print(lda.print_topics(20))
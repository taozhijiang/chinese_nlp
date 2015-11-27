#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Labeled Latent Dirichlet Allocation
# This code is available under the MIT License.
# (c)2010 Nakatani Shuyo / Cybozu Labs Inc.
# refer to Ramage+, Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora(EMNLP2009)

import os
import sys, re, numpy
import jieba
import hanzi_util
import pickle

DATA_DIR   = os.getcwd() + "/../data_dir/" 
STOP_FILE  = DATA_DIR + "stopwords.txt"
LABEL_TRAIN_FILE = DATA_DIR + "label_jd.txt"
#LABEL_TRAIN_FILE = DATA_DIR + "label_rrd.txt"

stop_words = []
dump_data = []

if not os.path.exists("./dump.dat"):
	with open(STOP_FILE, 'r') as fin:
		stop_words = []
		for line in fin:
			line = line.strip()
			stop_words.append(line)

			
	#stop
	fp = open("./dump.dat",'wb', -1)
	dump_data = []
	dump_data.append(stop_words)
	pickle.dump(dump_data, fp, -1)
	fp.close()
	
else:
	fp = open("./dump.dat",'rb')
	dump_data = pickle.load(fp)
	stop_words = dump_data[0]
	fp.close()

def load_corpus(filename):
    corpus = []
    labels = []
    labelmap = dict()
    with open(filename, 'r') as fin:
        for line in fin:
            line = line.strip()
            mt = re.match(r'\[(.+?)\](.+)', line)
            if mt:
                label = mt.group(1).split(',')
                for x in label: labelmap[x] = 1
                line = mt.group(2).strip()
            else:
                label = None
            #标签后的文本内容
            #太长的文本丢弃掉
            if(len(line) > 512):
                continue
            line = jieba.cut(line, cut_all=False)
            doc = []
            for item in line:
                if item not in stop_words and hanzi_util.is_zhs(item):
                    doc.append(item)
            if len(doc)>0:
                corpus.append(doc)
                labels.append(label)
    
    return labelmap.keys(), corpus, labels

# document-topic (theta) 
# topic-word (lambda)     
    
class LLDA:
    def __init__(self, K, alpha, beta):
        #self.K = K
        self.alpha = alpha
        self.beta = beta

    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def complement_label(self, label):
        if not label: return numpy.ones(len(self.labelmap))
        vec = numpy.zeros(len(self.labelmap))
        vec[0] = 1.0
        for x in label: vec[self.labelmap[x]] = 1.0
        return vec

    def set_corpus(self, labelset, corpus, labels):
        #labelset.insert(0, "common")
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)

        self.vocas = []
        self.vocas_id = dict()
        self.labels = numpy.array([self.complement_label(label) for label in labels])
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]

        M = len(corpus)
        V = len(self.vocas)

        self.z_m_n = []
        self.n_m_z = numpy.zeros((M, self.K), dtype=int)
        self.n_z_t = numpy.zeros((self.K, V), dtype=int)
        self.n_z = numpy.zeros(self.K, dtype=int)

        for m, doc, label in zip(range(M), self.docs, self.labels):
            N_m = len(doc)
            #z_n = [label[x] for x in numpy.random.randint(len(label), size=N_m)]
            z_n = [numpy.random.multinomial(1, label / label.sum()).argmax() for x in range(N_m)]
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1

    def inference(self):
        V = len(self.vocas)
        for m, doc, label in zip(range(len(self.docs)), self.docs, self.labels):
            for n in range(len(doc)):
                t = doc[n]
                z = self.z_m_n[m][n]
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                denom_a = self.n_m_z[m].sum() + self.K * self.alpha
                denom_b = self.n_z_t.sum(axis=1) + V * self.beta
                p_z = label * (self.n_z_t[:, t] + self.beta) / denom_b * (self.n_m_z[m] + self.alpha) / denom_a
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_m_n[m][n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def phi(self):
        V = len(self.vocas)
        return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + V * self.beta)

    def theta(self):
        """document-topic distribution"""
        n_alpha = self.n_m_z + self.labels * self.alpha
        return n_alpha / n_alpha.sum(axis=1)[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.phi()
        thetas = self.theta()

        log_per = N = 0
        for doc, theta in zip(docs, thetas):
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

def main():

    labelset, corpus, labels = load_corpus(LABEL_TRAIN_FILE)
    print("哈哈")
    print(labelset)

    if not os.path.exists("llda.dat"):
        llda = LLDA(K=len(labelset), alpha=0.001, beta=0.001)
        llda.set_corpus(labelset, corpus, labels)
        print ("M=%d, V=%d, L=%d, K=%d" % (len(corpus), len(llda.vocas), len(labelset), len(labelset)))

        for i in range(100):
            print("-- %d " % (i + 1))
            llda.inference()
            
        with open("llda.dat", 'wb') as fp:
            pickle.dump(llda, fp, -1)
    else:
        print("loading llda...")
        with open("llda.dat", 'rb') as fp:
            llda = pickle.load(fp)
        
    #困惑度 #通常情况下，困惑度越低，说明模型产生文档的能力越高，模型的推广性也就越好，通过观测困惑度来调整K取值
    print ("perplexity : %.4f" % llda.perplexity())

    phi = llda.phi()
    theta = llda.theta()
    for k, label in enumerate(labelset):
        print ("\n-- label %d : %s" % (k, label))
        for w in numpy.argsort(-phi[k])[:10]:
            #print("%d~%s" %(k, w))
            print ("%s: %.4f" % (llda.vocas[w], phi[k,w]))
            
    test_str = "如何变更手机号码？"
    line = jieba.cut(test_str.strip(), cut_all=False)
    obj = []
    for item in line:
        if item not in stop_words and hanzi_util.is_zhs(item):
            obj.append(item)
    #print(llda.phi())
    #print(llda.theta())
    print(len(phi))
    print(len(theta[0]))
    for k, label in enumerate(labelset):
        print(theta[llda.term_to_id(obj[0]),k])

if __name__ == "__main__":
    main()

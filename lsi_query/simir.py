#!/usr/bin/python3
import jieba
import hanzi_util
import pickle
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

documents = []
texts = []
stop_words = set()

STOP_FILE = "stopwords.txt"
doc_name  = "wo.txt"
DUMPFILE  = "dump.dat"

if __name__ == "__main__":


    if not os.path.exists(DUMPFILE):

        with open(STOP_FILE, 'r') as fin:
            for line in fin:
                line = line.strip()
                stop_words.add(line)

        line_num = 0
        frequency = {}
        with open(doc_name) as fin:
            for line in fin:
                line = line.strip()
                if len(line) > 50: continue
                if len(line) < 4: continue
                line_num += 1
                if not line_num % 5000: print("CURR:%d"%(line_num))
                seg_list = list(jieba.cut(line, cut_all=False))
                while '' in seg_list:
                    seg_list.remove('')
                line_t = [ x for x in seg_list if x not in stop_words and hanzi_util.is_zhs(x)]
                for token in line_t:
                    if token in frequency:
                        frequency[token] += 1
                    else:
                        frequency[token] = 1
                if not line_t: continue
                texts.append(line_t)
                documents.append(line)

        #texts = [[token for token in text if frequency[token] > 1]
        #          for text in texts]

        del frequency
        dictionary = corpora.Dictionary(texts)
        k_value = len(dictionary) * 0.25
        print("字典大小:%d" %(len(dictionary)))
        print("k值：%d" %(int(k_value)))


        # 文档->ID
        corpus = [dictionary.doc2bow(text) for text in texts]

        tfidf = models.TfidfModel(corpus)
        tfidf_corpus = tfidf[corpus]

        # num_topics，起到降维的作用，推荐参数为 200–500
        lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=int(k_value))
        index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it

        del texts

        # DUMP DATA:
        fp = open(DUMPFILE,'wb', -1)
        dump_data = []
        dump_data.append(index)
        dump_data.append(documents)
        dump_data.append(lsi)
        dump_data.append(dictionary)
        pickle.dump(dump_data, fp, -1)
        del dump_data


    else:

        print("加载索引信息！")
        fp = open(DUMPFILE,'rb')
        dump_data = pickle.load(fp)
        index = dump_data[0]
        documents = dump_data[1]
        lsi = dump_data[2]
        dictionary = dump_data[3]

    while True:
        while True:
            input_Q = input("Q: ").strip()
            if input_Q:
                break
        seg_list = list(jieba.cut(input_Q, cut_all=False)) #False
        while '' in seg_list:
            seg_list.remove('')
        new_vec_bow = dictionary.doc2bow(seg_list)
        new_vec_lsi = lsi[new_vec_bow]   # convert the query to LSI space
        sims = index[new_vec_lsi]        # perform a similarity query against the corpus
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        print("Query: %s" %(input_Q))
        print(seg_list)
        for item in sims[:7]:
            print("%f <-> %s" %(item[1], documents[item[0]]))

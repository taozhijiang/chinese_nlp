#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import numpy as np
import sys
from hmmlearn import hmm
import hanzi_util
import copy
import math

import pickle

DATA_DIR   = os.getcwd() + "/../data_dir/" 
CHAR_FILE  = DATA_DIR + "hcutf8.txt"
DICT_FILE  = DATA_DIR + "jieba_dict.txt"
TRAIN_FILE = DATA_DIR + "icwb2-data/training/msr_pku_training.utf8"

# BEMS from jieba的经验值
P_START = [0.7689828525554734, 0.0, 0.0, 0.2310171474445266 ];

enums = ['B', 'E', 'M', 'S']
enumo = []
seg_dict = {}

# Ci->Cj  转移矩阵
count_trans = {'B':{'B':0, 'E':0, 'M':0, 'S':0}, 'E':{'B':0, 'E':0, 'M':0, 'S':0}, 'M':{'B':0, 'E':0, 'M':0, 'S':0}, 'S':{'B':0, 'E':0, 'M':0, 'S':0}}
P_transMatrix = {'B':{'B':0, 'E':0, 'M':0, 'S':0}, 'E':{'B':0, 'E':0, 'M':0, 'S':0}, 'M':{'B':0, 'E':0, 'M':0, 'S':0}, 'S':{'B':0, 'E':0, 'M':0, 'S':0}}
# Ci->Oj  混淆矩阵
count_mixed = {'B':{}, 'E':{}, 'M':{}, 'S':{}}
P_mixedMatrix = {'B':{}, 'E':{}, 'M':{}, 'S':{}}

#建立中文的字符映射（当前还没考虑对繁体中文的处理）
def st_build_enumo(charfile):
	global enumo
	enumo = []
	fin = open(charfile,"r")
	for line in fin:
		line = line.strip()
		if not line or line[0] == '#':
			continue
		enumo.append(line[0])
		continue
	return

def st_build_dict(dict_file):
	global seg_dict
	seg_dict = {}
	fin = open(dict_file,"r")
	for line in fin:
		line = line.strip()
		if not line or line[0] == '#':
			continue
		line_t = line.split()
		if not hanzi_util.is_zhs(line_t[0]):
			print("SKIP:%s"%(line_t[0]))
			continue
		if(len(line_t[0]) == 1):
			if line_t[0] not in seg_dict.keys():
				seg_dict[line_t[0]] = [line_t[0]]
			else:
				print("EEEEEEEEEEEEEEEEEEEEEE1")
				seg_dict[line_t[0]].append(line_t[0])
				return
		else:
			chr = line_t[0][0]
			if chr not in seg_dict.keys():
				seg_dict[chr] = [line_t[0]]
			else:
				seg_dict[chr].append(line_t[0])


def st_trainMatrix(trainfile):
	with open(trainfile) as fin:
		for line in fin:
			line = line.strip()
			line_items = line.split()
			for item in line_items:
				if hanzi_util.is_terminator(item) or ( len(item) ==1 and hanzi_util.is_punct(item) ):
					line_items.remove(item);
			# whether exists elements
			if not line_items:
				continue
			# BEMS encode
			# line_hits  <-->  line_items
			# 进行字符和处理结果的对应
			line_hits = []	# every char status
			for i_index in range(len(line_items)):
				if len(line_items[i_index]) == 1:
					line_hits += 'S'
				else:
					for j_index in range(len(line_items[i_index])):
						if j_index == 0:
							line_hits += 'B'
						elif j_index == len(line_items[i_index]) - 1:
							line_hits += 'E'
						else:
							line_hits += 'M'
			if len(''.join(line_items)) != len(line_hits):
				print("EEEEEEE %d<->%d" %(len(''.join(line_items)),len(line_hits)));
			#print(''.join(line_items))
			#print(line_hits)
			line_items = ''.join(line_items)

			for i in range(len(line_hits)-1):
				# for calc trans matrix P[I][J]
				count_trans[line_hits[i]][line_hits[i+1]] += 1
			for i in range(len(line_hits)-1):
				# for calc mixed_matrix 
				if line_items[i] not in count_mixed[line_hits[i]].keys():
					count_mixed[line_hits[i]][line_items[i]] = 1
				else:
					count_mixed[line_hits[i]][line_items[i]] += 1

	for (k_i, v_i) in count_trans.items():
		count = sum(v_i.values())
		for (k_j, v_j) in v_i.items():
			P_transMatrix[k_i][k_j] = v_j / count
    
	for (k_i, v_i) in count_mixed.items():
		for item in enumo:
			if item not in v_i.keys():
				count_mixed[k_i][item] = 1	#针对没有出现的词，将其出现频次设置为1

	for (k_i, v_i) in count_mixed.items():
		count = sum(v_i.values())
		for (k_j, v_j) in v_i.items():
			P_mixedMatrix[k_i][k_j] = (v_j +1) / count #添加1进行平滑	

	return 

#针对每一次匹配，将结果保留到这个list中，单线程操作没有问题，不用数据保护
case_es = []
def sub_scan_str(str, str_list, i):
	global case_es
	str_len = len(str_list)
	if i >= str_len:
		return
	if str_len == 1:
		str_list[0] += 'S'
		case_es.append(str_list)
		return
	elif i == str_len - 1:
		str_list[i] += 'S'
		case_es.append(str_list)
		return
	else:
		chr = str_list[i]
		# 字典没有收录的词，直接分为1个词。此处最好做等级增补
		if chr not in seg_dict.keys():
			print("EEEEEEEEEEEEEEE2 %c " %(chr))
			str_list_t = copy.deepcopy(str_list)
			str_list_t[i] += 'S'
			return sub_scan_str(str, str_list_t, i + 1)
		else:
			j = str_len
			while j > i:
				if str[i:j] in seg_dict[chr]:
					str_list_t = copy.deepcopy(str_list)
					if j - i == 1:
						str_list_t[i] += 'S'
						if j == str_len:
							case_es.append(str_list_t)
							return
						else:
							sub_scan_str(str, str_list_t, j)
					elif j - i == 2:
						str_list_t[i] += 'B'
						str_list_t[i+1] += 'E'
						if j == str_len:
							case_es.append(str_list_t)
						else:
							sub_scan_str(str, str_list_t, j)
					else:
						str_list_t[i] += 'B'
						for z in range(i+1, j - 1):
							str_list_t[z] += 'M'
						str_list_t[j-1] += 'E'
						if j == str_len:
							case_es.append(str_list_t)
						else:
							sub_scan_str(str, str_list_t, j)
				j = j - 1
				
	return

# 此处作为评估作用，选出各种切分中最大概率作为分词结果
def match_str(str, model_t):
	global case_es
	case_es = []
	line_s = list(str)
	sub_scan_str(str, line_s, 0)
	case_pri = [0 for i in range(len(case_es))]
	for i in range(len(case_es)):
		#print("%d-%s"%(i,case_es[i]))
		pri = 0
		for i_t in case_es[i]:
			pri += (-math.log(p_mixed_matrix[enums.index(i_t[1])][enumo.index(i_t[0])]))
		case_pri[i] = pri
	for i in range(len(case_es)): 
		print("%f-%s" %(case_pri[i], case_es[i]))
	good_index = case_pri.index(min(case_pri))
	return (good_index, case_es[good_index])

def show_result(str_list):
	for i in range(len(str_list)):
		print(str_list[i][0], end='')
		if str_list[i][1] == 'E' or str_list[i][1] == 'S':
			print('-', end='')


if __name__ == "__main__":
	print("HELLO START");
	
	dump_data = [];

	if not os.path.exists("./dump.dat"):
		# get trans_Matrix and mixed_Matrix
		st_build_enumo(CHAR_FILE)
		print("CHARSIZE:%d"%len(enumo))
		st_build_dict(DICT_FILE)
		print("DICTSIZE:%d"%len(seg_dict))
		st_trainMatrix(TRAIN_FILE)
		print("Build trainMatrix Done!");
		
		p_trans_matrix = matrix = [[0 for col in range(4)] for row in range(4)]  
		for i in range(4):
			for j in range(4):
				p_trans_matrix[i][j] = P_transMatrix[enums[i]][enums[j]]

		#print(len(enumo)) 4677
		p_mixed_matrix = [[0 for col in range(len(enumo))] for row in range(4)]  
		for i in range(4):
			for j in range(len(enumo)):
				#print('CHAR:%s'%(enumo[j]))
				p_mixed_matrix[i][j] = P_mixedMatrix[enums[i]][enumo[j]]
		
		fp = open("./dump.dat",'wb', -1)
		dump_data.append(P_START)
		dump_data.append(p_trans_matrix)
		dump_data.append(p_mixed_matrix)
		dump_data.append(enumo)
		dump_data.append(seg_dict)
		pickle.dump(dump_data, fp, -1)
	else:
		fp = open("./dump.dat",'rb')
		dump_data = pickle.load(fp)
		P_START = dump_data[0]
		p_trans_matrix = dump_data[1]
		p_mixed_matrix = dump_data[2]
		enumo = dump_data[3]
		seg_dict = dump_data[4]

	model = hmm.MultinomialHMM(n_components=4)
	model.startprob_ = np.array(P_START)			#起始概率
	model.transmat_ = np.array(p_trans_matrix)		#转移矩阵
	model.emissionprob_ = np.array(p_mixed_matrix)	#混淆矩阵


	#test_str = "汉字笔顺标准由国家语言文字工作委员会标准化工作委员会制定叫做现代汉语通用字笔顺规范"
	#test_str = "研究生命"
	test_str = "掌握字符间的分和"
	(index, result) = match_str(test_str, model)
	show_result(result)
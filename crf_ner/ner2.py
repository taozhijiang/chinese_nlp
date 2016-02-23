#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import jieba

import os
import sys
import subprocess

IN_FILE = "199801.txt"
IN_FILE_NER = "199801.ner"
IN2_FILE_NER = "train_utf8.ner"
TRAIN_FILE = "ner.train"
MODEL_FILE = "ner.model"
TEST_FILE  = "ner.test"

src_tag = ['nr', 'ns', 'nt', 'nz']
dest2_tag = ['PER', 'LOC', 'ORG', 'MISC']

#w[-2]=He, w[-1]=reckons, w[0]=the, w[1]=current, w[2]=account
#w[-1]|w[0]=reckons|the, w[0]|w[1]=the|current

def prep_train_file1():
    with open(IN_FILE) as fin, open(IN_FILE+".tmp", "w") as ftmp:
        while True:
            try:
                line = fin.readline()
            except:
                print("READ ERROR:%d"%(LINE_NU))

            if not line:
                print("PROCESS DONE!")
                break

            line = line.strip()
            line_ls = line.split()
            line_len = len(line_ls)

            index = 0
            pre_tag = ''
            line_ret = []
            while index < line_len:
                if line_ls[index][0] == '[':
                    # 暂不处理[]嵌套的标注
                    #tmp_item = item.split('/')[0][1:]
                    #index_t = index + 1
                    #while index_t < line_len and line_ls[index_t][-3] != ']':
                    #    tmp_item += line_ls[index_t].split('/')[0]
                    #    index_t += 1
                    #tmp_item += line_ls[index_t].split('/')[0]
                    #print(line_ls[index:index_t+1])
                    #print("----" + tmp_item)
                    #tmp_item += '/' + line_ls[index_t][-2:]
                    #line_ret.append(tmp_item)
                    #index = index_t
                    #index += 1
                    #continue
                    line_ls[index] = line_ls[index][1:] 
                if line_ls[index][0] != ']' and ']' in line_ls[index]:
                    line_ls[index] = line_ls[index][:line_ls[index].index(']')]

                item = line_ls[index]
                if item[-2:] in src_tag:
                    if pre_tag and pre_tag == item[-2:]:
                        line_ret[-1] =  line_ret[-1][:-3] + item
                    else:
                        line_ret.append(item)
                    pre_tag = item[-2:]
                    index += 1
                    continue

                # Default
                line_ret.append(line_ls[index])
                pre_tag = ''
                index += 1
                continue

            for item in line_ret:
                ftmp.write(item + ' ')
            ftmp.write('\n')

    # STAGE2
    with open(IN_FILE+".tmp", "r") as fin, open(IN_FILE_NER, "w") as fner:
        while True:
            try:
                line = fin.readline()
            except:
                print("READ ERROR:%d"%(LINE_NU))

            if not line:
                print("PROCESS DONE!")
                break

            line_ls = line.strip().split()
            for item in line_ls:
                if item[-2:] not in src_tag:
                    for i_w in item[:-3]:
                        fner.write("%c N\n"%(i_w))
                else:
                    if item[-2:] == 'nr':
                        fner.write('%c B-PER\n' %(item[0]))
                        for i_wc in item[1:-3]:
                            fner.write('%c I-PER\n' %(i_wc))
                    elif item[-2:] == 'ns':
                        fner.write('%c B-LOC\n' %(item[0]))
                        for i_wc in item[1:-3]:
                            fner.write('%c I-LOC\n' %(i_wc))
                    elif item[-2:] == 'nt':
                        fner.write('%c B-ORG\n' %(item[0]))
                        for i_wc in item[1:-3]:
                            fner.write('%c I-ORG\n' %(i_wc))
                    elif item[-2:] == 'nz':
                        fner.write('%c B-MISC\n' %(item[0]))
                        for i_wc in item[1:-3]:
                            fner.write('%c I-MISC\n' %(i_wc))


def train_model():
    prep_train_file1()

    c1 = []
    t1 = []
    with open(IN_FILE_NER, 'r') as fin:
        while True:
            try:
                line = fin.readline()
            except:
                print("READ ERROR:%d"%(LINE_NU))

            if not line:
                print("PROCESS DONE!")
                break
            line = line.strip().split()
            if not line or len(line) != 2:
                continue
            c1.append(line[0])
            t1.append(line[1])

    c2 = []
    t2 = []
    with open(IN2_FILE_NER, 'r') as fin:
        while True:
            try:
                line = fin.readline()
            except:
                print("READ ERROR:%d"%(LINE_NU))

            if not line:
                print("PROCESS DONE!")
                break
            line = line.strip().split()
            if not line or len(line) != 2:
                continue
            c2.append(line[0])
            t2.append(line[1])

    len_1 = len(c1)
    len_2 = len(c2)

    str = ''
    with open(TRAIN_FILE,'w') as fout:
        for i in range(2, len_1-2):
            str = t1[i] + "\t"
            str += "w[-2]=" + c1[i-2] + '\t'
            str += "w[-1]=" + c1[i-1] + '\t'
            str += "w[-1]|w[0]=" + c1[i-1] + '|' + c1[i] + '\t'
            str += "w[0]=" + c1[i] + '\t'
            str += "w[1]=" + c1[i+1] + '\t'
            str += "w[0]|w[1]=" + c1[i] + '|' + c1[i+1] + '\t'
            str += "w[2]=" + c1[i+2] + '\t'
            str += "\n"
            fout.write(str)

        for i in range(2, len_2-2):
            str = t2[i] + "\t"
            str += "w[-2]=" + c2[i-2] + '\t'
            str += "w[-1]=" + c2[i-1] + '\t'
            str += "w[-1]|w[0]=" + c2[i-1] + '|' + c2[i] + '\t'
            str += "w[0]=" + c2[i] + '\t'
            str += "w[1]=" + c2[i+1] + '\t'
            str += "w[0]|w[1]=" + c2[i] + '|' + c2[i+1] + '\t'
            str += "w[2]=" + c2[i+2] + '\t'
            str += "\n"
            fout.write(str)

    print("TRAIN THE MODEL...")
    CMD = "./crfsuite learn -p max_iterations=500 -m %s %s" %(MODEL_FILE, TRAIN_FILE)
    print("CMD:%s"%(CMD))
    os.system(CMD)

def tag_me(str_test):
    if not str_test or len(str_test) < 5:
        return None

    with open(TEST_FILE,'w') as fout:
        for i in range(2, len(str_test)-2):
            str = "N" + "\t"
            str += "w[-2]=" + str_test[i-2] + '\t'
            str += "w[-1]=" + str_test[i-1] + '\t'
            str += "w[-1]|w[0]=" + str_test[i-1] + '|' + str_test[i] + '\t'
            str += "w[0]=" + str_test[i] + '\t'
            str += "w[1]=" + str_test[i+1] + '\t'
            str += "w[0]|w[1]=" + str_test[i] + '|' + str_test[i+1] + '\t'
            str += "w[2]=" + str_test[i+2] + '\t'
            str += "\n"
            fout.write(str)
    CMD = "./crfsuite tag -m %s %s" %(MODEL_FILE, TEST_FILE)
    print("CMD:%s"%(CMD))
    p = subprocess.Popen(CMD.split(), stdout=subprocess.PIPE)
    (stdout, stderr) = p.communicate()

    if p.returncode != 0:
        print(stderr)
        raise OSError('%s command failed!' %(CMD))

    return stdout.decode('utf-8').strip().split('\n')


if __name__ == "__main__":

    if not os.path.exists(MODEL_FILE):
        train_model()

    str = "春节前夕，中共中央总书记、国家主席、中央军委主席习近平等党和国家领导人分别看望或委托有关方面负责同志看望了江泽民、胡锦涛、李鹏、万里、乔石、朱镕基、李瑞环、吴邦国、温家宝、贾庆林、宋平、尉健行、李岚清、曾庆红、吴官正、李长春、罗干、贺国强和张劲夫、田纪云、迟浩田、姜春云、钱其琛、王乐泉、王兆国、回良玉、刘淇、吴仪、郭伯雄、曹刚川、曾培炎、王刚、王汉斌、张震、何勇、王丙乾、邹家华、王光英、布赫、铁木尔·达瓦买提、彭珮云、周光召、曹志、李铁映、司马义·艾买提、何鲁丽、丁石孙、成思危、许嘉璐、蒋正华、顾秀莲、热地、盛华仁、路甬祥、乌云其木格、华建敏、陈至立、周铁农、司马义·铁力瓦尔地、蒋树声、桑国卫、唐家璇、梁光烈、戴秉国、肖扬、韩杼滨、贾春旺、叶选平、杨汝岱、任建新、宋健、钱正英、孙孚凌、万国权、胡启立、陈锦华、赵南起、毛致用、王忠禹、李贵鲜、张思卿、罗豪才、张克辉、郝建秀、徐匡迪、张怀西、李蒙、廖晖、白立忱、陈奎元、阿不来提·阿不都热西提、李兆焯、黄孟复、张梅颖、张榕明、钱运录、孙家正、李金华、郑万通、邓朴方、厉无畏、陈宗兴、王志珍等老同志，向老同志们致以诚挚的节日问候，衷心祝愿老同志们新春愉快、健康长寿。"
    result = tag_me(str)

    for i in range(len(result)):
        if result[i] != 'N':
            print("%s%s "%(str[i+2], result[i]), end="")
        else:
            print("%s"%(str[i+2]), end="")
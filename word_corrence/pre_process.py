#!/usr/bin/python3
import os
import jieba
import hanzi_util

DATA_DIR = os.getcwd() + "/../data_dir/ClassFile/" 
STOP_FILE = os.getcwd() + "/../data_dir/stopwords.txt"

stop_words = []
with open(STOP_FILE, 'r') as fin:
    for line in fin:
        line = line.strip()
        if not line or line[0] == '#': continue
        stop_words.append(line)
print("STOP WORD SIZE:%d\n" %(len(stop_words)))

#统计总的标签数目
for parent,dirname,filenames in os.walk(DATA_DIR):
    for filename in filenames:
        if filename[-4:] != '.txt': continue
        if filename[-6:] == '_p.txt': continue
        tag_name = filename[:-4]
        line_num = 0
        with open(DATA_DIR+'/'+filename,'r') as fin, open(DATA_DIR+'/'+tag_name+'_p.txt','w') as fout:
            print('正在处理：%s'%(filename))
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
                if not line_num % 1000 : print('LINE:%d'%(line_num))
                line = line.strip()
                line_t = jieba.cut(line, cut_all=False)
                objs = []
                for item in line_t:
                    if item not in stop_words and hanzi_util.is_zhs(item):
                        if item not in objs: 
                            objs.append(item)
                if not len(objs): continue
                line = ' '.join(objs)+'\n'
                #print(line, end='')
                fout.write(line)

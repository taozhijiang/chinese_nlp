#!/usr/bin/python3

import pickle

#用于将跑出来的临时数据保存成文本的格式，方便C程序读取调用

CURRENT_VER = 4

dump_file_data = "./dump_data.dat_v%d"%(CURRENT_VER)
dump_file_class = "./dump_class.dat_v%d"%(CURRENT_VER)

dump_file_cpp = "./dump_cpp.dat_v%d"%(CURRENT_VER)

print("LOADING DATA....")
with open(dump_file_data,'rb', -1) as fp:
    dump_data = pickle.load(fp)
    train_word_id = dump_data[0]
    train_tags = dump_data[1]
    stop_words = dump_data[2]
    white_words = dump_data[3]
    train_info = dump_data[4]
    sorted_word_scores = dump_data[5]
    del dump_data
    
with open(dump_file_cpp, 'w') as fout:

    # #@训练标签
    print('#@训练标签:\n')
    fout.write('#@训练标签:\n')
    for item in train_tags:
        fout.write("%d-%s\n"%(train_tags.index(item), item))
    fout.flush()
        
    # #@训练词表
    print('#@训练词表:\n')
    fout.write('#@训练词表:\n')
    for item in train_word_id:
        fout.write("%d-%s\n"%(train_word_id.index(item), item))
    fout.flush()
        
    # #@卡方指数
    print('#@卡方指数:\n')
    fout.write('#@卡方指数:\n')
    for item in sorted_word_scores:
        fout.write('%d-%f\n'%(item))
    fout.flush()
    
    # #@训练集
    fout.write('#@训练集:\n')
    for tag in train_tags[1:]:
        fout.write('#$%s:\n'%(tag))
        for item in train_info[train_tags.index(tag)]:
            fout.write(repr(item)+'\n')
    fout.flush()
    
    print('DONE!')
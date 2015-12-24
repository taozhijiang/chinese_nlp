#include "header.hpp"
#include <iostream>
#include <fstream>

using namespace std;

P_Jieba jieba_initialize(void) 
{
    cout << "Initializing jieba_cut ..." << endl;
    static const char* DICT_PATH = "./source/dict/jieba.dict.utf8";
    static const char* HMM_PATH = "./source/dict/hmm_model.utf8";
    static const char* USER_DICT = "./source/dict/user.dict.utf8";

    // init will take a few seconds to load dicts.
    P_Jieba handle = NewJieba(DICT_PATH, HMM_PATH, USER_DICT); 

    return handle;
}

// Caller should free the storage
bool jieba_cut(P_Jieba jieba, string str, vector<std::string> &elems)
{
    if (!jieba)
        return false;

    char** words = Cut(jieba, str.c_str()); 
    elems.clear();

    char** x = words;
    while (x && *x) 
    {
        elems.push_back(*x);
        x++;
    }
    FreeWords(words);

    return true;
}

void jieba_close(P_Jieba jieba)
{
    if (! jieba)
        return;

    FreeJieba(jieba);
}

void jieba_test(void)
{
    string str = "广州恒大门将曾诚近日在接受广州竞赛频道《足球粤势力》节目专访时表示自己跟布冯一个类型";

    P_Jieba jieba = jieba_initialize();
    vector<std::string> store;
    if (jieba_cut(jieba, str, store))
    {
        cout << "测试语句：" << str << endl;
        cout << "分词结果：";
        for (int i=0; i<store.size(); i++)
        {
            cout << store[i] << "  ";
        }
            cout << endl;
    }
    else
    {
        cout << "CUT FAILED!" << endl;
    }

    return;
}


bool load_train_data(string filename, CLASS_DATA_STRUCT &cds)
{
    //initialize
    cds.train_tags.clear();
    cds.train_w_id.clear();
    cds.train_id_w.clear();
    cds.sorted_wscores.clear();
    cds.train_info.clear();
    cds.test_info.clear();

    ifstream fin(filename);
    string line;

    string curr_cat = "";
    int    curr_tag_id = 0;
    int    curr_index = 0;

    while (getline(fin, line))
    {
        if(!line.length())
            continue;

        if (line[0] == '#')
        {
            if ( line[1] != '@' && line[1] != '$')
                continue;
            
            if ( line[1] == '@')
            {
                if ( curr_cat.length() )
                    cout << curr_cat << "，结束!" << endl; 
                curr_cat = line.substr(2);
                curr_index = 0;
                cout << "DETECT_CAT:" << curr_cat << endl;
            }

            if ( line[1] == '$')
            {
                // 还需要去掉结尾的:
                curr_tag_id = 0;
                string curr_tag = line.substr(2, line.length()-3);
                vector< vector<int> > dummy;
                cout << "DETECT_TAG:" << curr_tag << endl;
                for (int i=0; i< cds.train_tags.size(); i++)
                {
                    if ( cds.train_tags[i] == curr_tag)
                    {
                        curr_tag_id = i;
                        cout << "FOUND ID:" << i << " for " << curr_tag << endl;
                        break;
                    }
                }
                if (curr_tag_id == 0)
                {
                    cerr << "ERROR TAG_ID not found!" << curr_tag << endl;
                    exit(-1);
                }

                cds.train_info[curr_tag_id] = dummy;
                cds.test_info[curr_tag_id] = dummy;

                curr_index = 0;
            }
            continue;
        }

        // 普通的数据
        // 训练标签  训练词表  卡方指数 训练集
        if ( curr_cat == "训练集:")
        {
            if ( curr_tag_id == 0)
            {
                cerr << "ERROR tag_id is 0" << endl;
                exit(-1);
            }
            if (line.length() <= 2)  // NULL []
                continue;

            if (line[0] != '[' || line[line.length() - 1] != ']') 
            {
                cerr << "ERROR LINE:" << line << endl;
                exit(-1);
            }

            line = line.substr(1, line.length()-2);
            vector<std::string> st;
            vector<int> st_n;
            split(line, ',', st);
            for (int i = 0; i< st.size(); i++)
                st_n.push_back(atoi(st[i].c_str()));

            // STORE IT!
            if (curr_index < 500 )
                cds.test_info[curr_tag_id].push_back(st_n);
            else
                cds.train_info[curr_tag_id].push_back(st_n);

            ++ curr_index;

        }
        else if ( curr_cat == "训练词表:")
        { 
            vector<string> tokens = split(line,'-');
            if ( atoi(tokens[0].c_str()) != curr_index)
            {
                cerr << "Error for mismatch: " << tokens[0].c_str() << "~" << curr_index << endl;
                exit(-1);
            }
            cds.train_w_id[tokens[1]] = curr_index;
            cds.train_id_w[curr_index] = tokens[1];
            ++ curr_index;
        }
        else if ( curr_cat == "卡方指数:")
        {
            vector<string> tokens = split(line,'-');
            cds.sorted_wscores.push_back(atoi(tokens[0].c_str()));
        }
        else if ( curr_cat == "训练标签:")
        {
            vector<string> tokens = split(line,'-');
            if ( atoi(tokens[0].c_str()) != curr_index)
            {
                cerr << "Error for mismatch: " << tokens[0].c_str() << "~" << curr_index << endl;
                exit(-1);
            }
            cds.train_tags.push_back(tokens[1]);
            ++ curr_index;
        }
        else if (curr_cat.length())
        {
            cerr << "ERROR CAT:" << curr_cat << endl;
            exit(-1);
        }
    }
    
    if (cds.train_w_id.size() != cds.sorted_wscores.size())
    {
        cerr << " WORD size mismatch!" << endl;
        exit(-1);
    }

    cout << "Initilaize OK!" << endl;
    cout << "TOTAL LENGTH INFO:" << endl;
    cout << "\tTRAIN_TAG:" << cds.train_tags.size() << endl;
    cout << "\tTRAIN_WORD_ID:" << cds.train_w_id.size() << endl;
    cout << "\tSORTED_WSCORES:" << cds.sorted_wscores.size() << endl;
    cout << "\tTRAIN&TEST_INFO:" << cds.train_info.size() << endl;
    for (int i = 1; i< cds.train_tags.size(); i++)
    {
        cout << "\t\t[TRAIN]" << cds.train_tags[i] << ":" << cds.train_info[i].size() << endl;
        cout << "\t\t[TEST ]" << cds.train_tags[i] << ":" << cds.test_info[i].size() << endl;
    }

    fin.close();

    return true; 
}

void utils_test(void)
{
    load_train_data("./dump_cpp.dat_v4", cds);
}

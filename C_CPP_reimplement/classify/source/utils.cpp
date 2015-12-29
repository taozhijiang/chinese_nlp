#include "header.hpp"
#include <iostream>
#include <fstream>
#include <string.h>
#include <dirent.h>  

#include <algorithm>

using namespace std;

P_Jieba jieba_initialize(void) 
{
    P_Jieba handle = NULL;

    cout << "Initializing jieba_cut ..." << endl;
    static const char* DICT_PATH = "./source/dict/jieba.dict.utf8";
    static const char* HMM_PATH = "./source/dict/hmm_model.utf8";
    static const char* USER_DICT = "./source/dict/user.dict.utf8";

    // init will take a few seconds to load dicts.
    handle = NewJieba(DICT_PATH, HMM_PATH, USER_DICT); 

    return handle;
}

// Caller should free the storage
bool jieba_cut(P_Jieba jieba, string str, vector<std::string> &elems)
{
    if (!jieba)
        return false;

    char** words =  Cut(jieba, str.c_str()); 
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

    return;
}

void jieba_test(void)
{
    string str = "伴随着申花必胜的呼喊，曼萨诺出现在了上海机场，成为了申花队的新任主教练。对于曼萨诺从国安来到申花，申花的球迷接受采访时说出了上面的话。";

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
    jieba_close(jieba);

    return;
}

static double calc_chi_sq(int cond_word_fd, int freq_w, long freq_t, long total_w)
{
    double n_ii = cond_word_fd;
    double n_oi = freq_t - n_ii;
    double n_io = freq_w - n_ii;
    double n_oo = total_w - n_io - n_io - n_ii;

    return ((n_ii*n_oo - n_io*n_oi)*(n_ii*n_oo - n_io*n_oi)) /
                ((n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo));
}

bool prep_train_data(CLASS_DATA_STRUCT &cds, string dirname)
{
    struct dirent* ent = NULL;  
    DIR *pDir = opendir(dirname.c_str());
    const char *data_subfix = ".txt";

    if(!pDir)
    {
        cerr << "Open Dir failed: " << dirname << endl;
        return false;
    } 
    
    cds.train_tags.clear();
    cds.sorted_wscores.clear();
    cds.train_w_id.clear();
    cds.train_id_w.clear();
    cds.train_w.clear();
    cds.train_info.clear();
    cds.test_info.clear();

    cds.train_tags.push_back("NULL");
    while( (ent = readdir(pDir)) != NULL)
    {
        if(ent->d_type == DT_REG)
        {
            //cout << ent->d_name << endl;
            if(!strcmp((ent->d_name + strlen(ent->d_name) -4), data_subfix) )
            {
                string tag_name = ent->d_name;
                tag_name = tag_name.substr(0, tag_name.size() - 4);
                cout << "GETTAG: " << tag_name << endl;
                cds.train_tags.push_back(tag_name);
            }
        }
    }
    closedir(pDir);
    cds.data_path = string(dirname);


    map<int, int> word_fd;
    map<int, map<int, int> > cond_word_fd;
    map<int, int> tmp_word_fd;

    map<int, long> total_tag_count;
    long  total_w_count = 0;
    long  tmp_for_check = 0;

    map<int, int> :: iterator it_ii;

    for(int i = 1; i < cds.train_tags.size(); ++i)
    {
        tmp_word_fd.clear();
        process_train_file(cds, i, word_fd, tmp_word_fd);
        cond_word_fd[i] = tmp_word_fd;

        for(it_ii = tmp_word_fd.begin(); it_ii != tmp_word_fd.end(); ++it_ii)
            total_tag_count[i] += it_ii->second;

        tmp_for_check += total_tag_count[i];
        cout << "TAG: " << cds.train_tags[i] << ", word freq :" <<  total_tag_count[i] << endl ;
    }

    for(it_ii = word_fd.begin(); it_ii != word_fd.end(); ++ it_ii)
        total_w_count += it_ii->second;

    if(total_w_count != tmp_for_check)
    {
        cerr << "TOTAL COUNT DISMATCH:" << tmp_for_check << "~" << total_w_count << endl;
        exit(-1);
    }
    else
        cout << "TOTAL FREQ COUNT:" << total_w_count << endl;

    // 卡方分布
    // 方检验存在所谓的 低频词缺陷，即低频词可能会有很高的卡方值。
    // 统计文档中是否出现词 t ，却不管 t 在该文档中出现了几次
   
    std::vector<std::pair<double,int>> w_scores;    // value-key for value autosort
    int word_id = 0;
    int tag_id = 0;
    double score = 0;

    cout << "TRIM LOW FREQ WORD!" << endl;
    for(it_ii = word_fd.begin(); it_ii != word_fd.end(); ++ it_ii)
    {
        word_id = it_ii->first;
        score = 0;
        if(word_fd[word_id] <= 5)
        {
            //cout << "SKIP:" << cds.train_id_w[word_id] << ", FREQ:" << word_fd[word_id] << endl;
            cout << ".";
            continue;
        }
        for(tag_id = 1; tag_id < cds.train_tags.size(); ++ tag_id)
        {
            score += total_w_count*calc_chi_sq(cond_word_fd[tag_id][word_id], word_fd[word_id],
                total_tag_count[tag_id], total_w_count);
        }

        w_scores.push_back(std::pair<double, int>(score, word_id));
       //cout << cds.train_id_w[word_id] << ":" << scores[word_id] << endl;
    }
    cout << endl;

    sort(w_scores.begin(), w_scores.end());

    for(int i = w_scores.size() - 1; i>=0; --i)
    {
        //cout << cds.train_id_w[w_scores[i].second] << ":" << w_scores[i].first << endl;
        cds.sorted_wscores.push_back(w_scores[i].second);
    }

    return true;
}

static inline bool is_zhs_UTF8(const char* str);
static bool process_train_file(CLASS_DATA_STRUCT &cds, int tag_id, 
    map<int, int> &word_fd, map<int, int> &tmp_word_fd)
{

    string tagname = cds.train_tags[tag_id];
    string filepath = cds.data_path + tagname + ".txt";

    cout << "PROCESSING:" << filepath << endl;

    ifstream fin(filepath);
    fin.ignore();
    string line;
    unsigned long line_num = 0;
    vector<std::string> cut_store;
    vector<std::string> cut_lite;
    vector<int> cut_lite_id;

    while (getline(fin, line))
    {
        if(!line.length())
            continue;

        ++ line_num;
        if(! (line_num % 10000))
            cout << "LINE:" << line_num << endl;

        if(! jieba_cut(cds.jieba, line, cut_store))
            continue;

        cut_lite.clear();
        cut_lite_id.clear();
        for(int i = 0; i < cut_store.size(); i++)
        {
            if( is_zhs_UTF8(cut_store[i].c_str()) )
                cut_lite.push_back(cut_store[i]);
        }

        int word_id = -1;    
        for(int i = 0; i < cut_lite.size(); i++)
        {
            auto word_num =  cds.train_w_id.find(cut_lite[i]);    
            if (word_num == cds.train_w_id.end() )
            {
                //添加入词表
                cds.train_w.push_back(cut_lite[i]);
                word_id = cds.train_w.size() - 1;   //词ID
                //cout << "添加词:" << cut_lite[i] << ", ID=" << word_id << endl;
                cds.train_w_id[cut_lite[i]] = word_id;
                cds.train_id_w[word_id] = cut_lite[i];
            }
            else
                word_id = word_num->second;

            cut_lite_id.push_back(word_id);
        }

        //保留作为训练数据和测试数据
        if(!cut_lite_id.empty())
        {
            if((line_num % 30) == 0)    // 4%
                cds.test_info[tag_id].push_back(cut_lite_id);
            else
                cds.train_info[tag_id].push_back(cut_lite_id);
        }
            
        // 词频统计
        // 由于是卡方统计，不计数，只表示是否出现，所以前面的内容还需要去重复处理

        //cout << "BEFORE" << cut_lite_id.size() << endl;
        sort(cut_lite_id.begin(), cut_lite_id.end());
        auto last = unique(cut_lite_id.begin(), cut_lite_id.end());
        cut_lite_id.erase(last, cut_lite_id.end());
        //cout <<  "AFTER" << cut_lite_id.size() << endl;

        for(int i = 0; i < cut_lite_id.size(); ++i)
        {
            word_id = cut_lite_id[i];
            word_fd[word_id] += 1;
            tmp_word_fd[word_id] += 1;
        }

    }

    cout << "TOTAL WORD SIZE: " << cds.train_w.size() << endl;
    cout << "DOC SIZE: " << cds.train_info[tag_id].size() << "/" << cds.test_info[tag_id].size() << endl;

    return true;
}

/*
    00000000 -- 0000007F:   0xxxxxxx
    00000080 -- 000007FF:   110xxxxx 10xxxxxx
    00000800 -- 0000FFFF:   1110xxxx 10xxxxxx 10xxxxxx
    00010000 -- 001FFFFF:   11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
*/
/**
 * 这里做一个简化，由于实现已经经过分词处理，所以这里认为：
 * 如果开头是ASCII字符，那么这个词就是英文单词
 * 如果UTF-8长度为1,那么可能是全角标点，或者单个中文词，同样返回false
 */
#define IS_IN_RANGE(c, f, l)    (((c) >= (f)) && ((c) <= (l)))
#define UTF8_CHAR_LEN( byte )   ((( 0xE5000000 >> (( byte >> 3 ) & 0x1e )) & 3 ) + 1)
static inline bool is_zhs_UTF8(const char* str)
{

        int len = 0, step = 0;
        unsigned int full_char = 0;

        if( !( str[0] & 0x80 )) //ASCII字符
            return false;

        // 单个字符或者全角标点
        if( UTF8_CHAR_LEN(str[0]) >= strlen(str) )
            return false;   

        return true;
}

bool save_train_data(CLASS_DATA_STRUCT &cds, string filename)
{
    ofstream fout(filename);
    int size_len = 0;
    
    cout << "#@训练标签:" << endl;
    fout << "#@训练标签:" << endl;
    size_len = cds.train_tags.size();
    for(int i =0 ; i<size_len; ++i)
        fout << i << "-" << cds.train_tags[i] << endl;
    fout.flush();

    cout << "#@训练词表:" << endl;
    fout << "#@训练词表:" << endl;
    size_len = cds.train_w.size();
    for(int i =0 ; i<size_len; ++i)
        fout << i << "-" << cds.train_w[i] << endl;
    fout.flush();

    cout << "#@卡方指数:" << endl;
    fout << "#@卡方指数:" << endl;
    size_len = cds.sorted_wscores.size();
    for(int i =0 ; i<size_len; ++i)
        fout << cds.sorted_wscores[i] << "-" << "0" << endl;
    fout.flush();

    cout << "#@训练集:" << endl;
    fout << "#@训练集:" << endl;
    for(int i =1; i<cds.train_tags.size(); ++i)
    {
        fout << "#$" << cds.train_tags[i] << ":" << endl;
        size_len = cds.train_info[i].size();
        for(int j=0; j<size_len; ++j)
        {
            fout << "[";
            for(int k=0; k<cds.train_info[i][j].size(); ++k)
            {
                fout << cds.train_info[i][j][k] <<"," ;
            }
            fout << "]" << endl;
        }
        fout.flush();
    }

    fout.flush();

    cout << "SAVE FINISHED!" << endl;

    fout.close();
}

bool load_train_data(CLASS_DATA_STRUCT &cds, string filename)
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

            // STORE IT!  ~4% test part
            if ( (curr_index % 30) == 0 )
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
            //vector<string> tokens = split(line,'-');
            int n_pos = line.find_first_of("-");

            if(n_pos != -1)
            {
                string str1 = line.substr(0, n_pos);
                string str2 = line.substr(n_pos + 1);
                
                if ( atoi(str1.c_str()) != curr_index)
                {
                    cerr << "Error for mismatch: " << str1.c_str() << "~" << curr_index << endl;
                    exit(-1);
                }
                cds.train_tags.push_back(str2);
                ++ curr_index;

            }
        }
        else if (curr_cat.length())
        {
            cerr << "ERROR CAT:" << curr_cat << endl;
            exit(-1);
        }
    }
    
    /* //剔除了部分词的卡方，所以这里肯定不相等了
    if (cds.train_w_id.size() != cds.sorted_wscores.size())
    {
        cerr << " WORD size mismatch!" << "train_w_id:"<< cds.train_w_id.size() 
            << "sorted_wscores:" << cds.sorted_wscores.size() << endl;
        exit(-1);
    }
    */

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
    load_train_data(cds, "./dump_cpp.dat_v4");
}

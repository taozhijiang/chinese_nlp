#include "header.hpp"
#include <iostream>
#include <fstream>
#include <string.h>
#include <dirent.h>  
#include <sstream>


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

#if 0
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
#endif

vector<string> &split(const string &s, char delim, vector<std::string> &elems) 
{
    stringstream ss(s);
    string item;
    while (std::getline(ss, item, delim)) 
    {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) 
{
    vector<string> elems;
    split(s, delim, elems);
    return elems;
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

        unique_vector(cut_lite_id);

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

bool unique_vector( vector<int> &vec)
{
    if(vec.empty())
        return false;

    //cout << "BEFORE" << cut_lite_id.size() << endl;
    sort(vec.begin(), vec.end());
    auto last = unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());
    //cout <<  "AFTER" << cut_lite_id.size() << endl;

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

/**
 * 由于MaxEntropy的训练速度非常的慢，所以这里保存最终的训练结果
 *
 *
    vector<std::string> train_tags;
    vector<std::string>   train_w;      //不断的push_back，索引放入下面两个map中
    map<int, int> useful_words; //word_id, BEST_N_word_index
    map<int, int> n_ft_index;
    vector<double> n_weight;
 */

bool save_train_data(CLASS_DATA_STRUCT &cds, string filename)
{
    ofstream fout(filename);
    int size_len = 0;
    map<int, int> ::iterator it;
    
    cout << "#@训练标签:" << endl;
    fout << "#@训练标签:" << endl;
    size_len = cds.train_tags.size();
    for(int i =0 ; i<size_len; ++i)
        fout << i << "$" << cds.train_tags[i] << endl;
    fout.flush();

    cout << "#@训练词表:" << endl;
    fout << "#@训练词表:" << endl;
    size_len = cds.train_w.size();
    for(int i =0 ; i<size_len; ++i)
        fout << i << "$" << cds.train_w[i] << endl;
    fout.flush();

    fout << "#@特征词表:" << endl;
    cout << "#@特征词表:" << endl;
    for(it = cds.useful_words.begin(); it != cds.useful_words.end(); ++it)
        fout << it->first << "$" << it->second << endl;
    fout.flush();

    fout << "#@特征映射表:" << endl;
    cout << "#@特征映射表:" << endl;
    for(it = cds.n_ft_index.begin(); it != cds.n_ft_index.end(); ++it)
        fout << it->first << "$" << it->second << endl;
    fout.flush();

    fout << "#@熵权值:" << endl;
    cout << "#@熵权值:" << endl;
    size_len = cds.n_weight.size();
    for(int i =0; i<size_len; ++i)
        fout << i << "$" << cds.n_weight[i] << endl;
    fout.flush();

    cout << "SAVE FINISHED!" << endl;

    fout.close();
}

bool load_train_data(CLASS_DATA_STRUCT &cds, string filename)
{
    //initialize
    cds.train_tags.clear();
    cds.train_w.clear();
    cds.train_w_id.clear();
    cds.train_id_w.clear();
    cds.useful_words.clear();
    cds.n_ft_index.clear();
    cds.n_weight.clear();


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
            if ( line[1] == '@')
            {
                if ( curr_cat.length() )
                    cout << curr_cat << "，结束!" << endl; 
                curr_cat = line.substr(2);
                curr_index = 0;
                cout << "DETECT_CAT:" << curr_cat << endl;
            }

            continue;
        }

        // 普通的数据
        // 训练标签  训练词表  特征词表 特征映射表 熵权值
         if ( curr_cat == "训练标签:")
        {
            //vector<string> tokens = split(line,'-');
            int n_pos = line.find_first_of("$");

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
        else if ( curr_cat == "训练词表:")
        { 
            vector<string> tokens = split(line,'$');
            if ( atoi(tokens[0].c_str()) != curr_index)
            {
                cerr << "Error for mismatch: " << tokens[0].c_str() << "~" << curr_index << endl;
                cerr << line << endl;
                exit(-1);
            }
            cds.train_w_id[tokens[1]] = curr_index;
            cds.train_id_w[curr_index] = tokens[1];
            ++ curr_index;
        }
        else if ( curr_cat == "特征词表:")
        {
            int n_pos = line.find_first_of("$");

            string str1 = line.substr(0, n_pos);
            string str2 = line.substr(n_pos + 1);

            cds.useful_words[atoi(str1.c_str())] = atoi(str2.c_str());

        }
        else if ( curr_cat == "特征映射表:")
        {

            int n_pos = line.find_first_of("$");

            string str1 = line.substr(0, n_pos);
            string str2 = line.substr(n_pos + 1);

            cds.n_ft_index[atoi(str1.c_str())] = atoi(str2.c_str());

        }
        else if ( curr_cat == "熵权值:")
        {
            int n_pos = line.find_first_of("$");

            string str1 = line.substr(0, n_pos);
            string str2 = line.substr(n_pos + 1);
            
            if ( atoi(str1.c_str()) != curr_index)
            {
                cerr << "Error for mismatch: " << str1.c_str() << "~" << curr_index << endl;
                exit(-1);
            }
            cds.n_weight.push_back(atof(str2.c_str()));
            ++ curr_index;

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
    cout << "\tUSEFUL_WORD:" << cds.useful_words.size() << endl;
    cout << "\tFT_INDEX:" << cds.n_ft_index.size() << endl;
    cout << "\tEntropyWeights:" << cds.n_weight.size() << endl;

    fin.close();

    return true; 
}

#if 0
void utils_test(void)
{
    load_train_data(cds, "./dump_cpp.dat_v4");
}
#endif

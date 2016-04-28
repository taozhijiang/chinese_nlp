#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <string>

#include <iomanip>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <float.h>


using namespace std;

string flag_a = "【";
string flag_b = "】";
string flag_c = "=】";
string flag_d = "～";

typedef struct _DATA_STRUCT
{
    map<string, map < string, set< string> > > RESULTS_WS;
    map<string, map < string, vector< string> > > RESULTS_SENT;      //标准例句
    map<string, set< string > > RESULTS_AIM;
    map<string, map < string, vector< string> > > RESULTS_EXT;       //扩展句
    //map<string, set< string> > RESULTS_RCPTS;    //需要反查的待选扩展
    map<string, map < string, set< string > > > RESULTS_RCPTS;    //需要反查的待选扩展
    map<string, map < string, vector< string > > > RESULTS_EXT_EXT;  //二级扩展句
    set<string> RESULTS_UNI;

} DATA_STRUCT, *P_DATA_STRUCT;

// trim from left
inline std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v")
{
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from right
inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v")
{
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from left & right
inline std::string& trim(std::string& s, const char* t = " \t\n\r\f\v")
{
    return ltrim(rtrim(s, t), t);
}


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

// 计算nil在str中出现的次数
static inline int string_counts(const string& str, const string& nil)
{
    if(!str.length() || ! nil.length())
        return -1;

    string::size_type cur_pos = 0;
    int count = 0;

    while((cur_pos = str.find(nil, cur_pos)) != string::npos)
    {
       count ++;
       cur_pos += nil.length();
    }

    return count;
}

void dump_data_stage1(string outfile, DATA_STRUCT& gdata)
{
    ofstream of(outfile);

    gdata.RESULTS_AIM.clear();
    gdata.RESULTS_UNI.clear();

    cout << "正在保存数据[dump_data_stage1]..." << endl;
    for( auto it=gdata.RESULTS_WS.begin(); it!=gdata.RESULTS_WS.end(); it++)
    {
        of << "[[" << it->first << "]]"<< endl; //多义词

        for(auto it2 = it->second.begin(); it2!= it->second.end(); ++it2)
        {
            of << "  " << it2->first << endl;   //意项

            of << "    ";
            for (auto it3 = it2->second.begin(); 
                    it3 != it2->second.end(); ++it3)
            {
                of << *it3 << " ";  // 同义词 
            }
            of << endl;

            // 标注例句
            of << "#标注例句" << endl;
            if(gdata.RESULTS_SENT.find(it->first) != gdata.RESULTS_SENT.end() 
                && gdata.RESULTS_SENT[it->first].find(it2->first) != gdata.RESULTS_SENT[it->first].end())
            {
                int sen_counts = gdata.RESULTS_SENT[it->first][it2->first].size();
                for(int i=0; i<sen_counts; ++i)
                    of << gdata.RESULTS_SENT[it->first][it2->first][i] << endl;
            }
            of << "#标注例句结束" << endl;

            //扩展例句
            of << "#扩展例句" << endl;
            if(gdata.RESULTS_EXT.find(it->first) != gdata.RESULTS_EXT.end() 
                && gdata.RESULTS_EXT[it->first].find(it2->first) != gdata.RESULTS_EXT[it->first].end())
            {
                int sen_counts = gdata.RESULTS_EXT[it->first][it2->first].size();
                for(int i=0; i<sen_counts; ++i)
                    of << gdata.RESULTS_EXT[it->first][it2->first][i] << endl;
            }
            of << "#扩展例句结束" << endl;

        }

        of.flush();
    }

    gdata.RESULTS_SENT.clear();
    gdata.RESULTS_EXT.clear();

    cout << "[dump_data_stage1]结束..." << endl;

    return;
}

void dump_data_stage2(string outfile, DATA_STRUCT& gdata)
{
    ofstream of(outfile);

    cout << "正在保存数据[dump_data_stage2]..." << endl;
    for( auto it=gdata.RESULTS_WS.begin(); it!=gdata.RESULTS_WS.end(); it++)
    {
        of << "[[" << it->first << "]]"<< endl; //多义词

        for(auto it2 = it->second.begin(); it2!= it->second.end(); ++it2)
        {
            of << "  " << it2->first << endl;   //意项

            of << "    ";
            for (auto it3 = it2->second.begin(); 
                    it3 != it2->second.end(); ++it3)
            {
                of << *it3 << " ";  // 同义词 
            }
            of << endl;

            //二级扩展句：
            of << "#二级扩展句" << endl;
            if(gdata.RESULTS_EXT_EXT.find(it->first) != gdata.RESULTS_EXT_EXT.end() 
                && gdata.RESULTS_EXT_EXT[it->first].find(it2->first) != gdata.RESULTS_EXT_EXT[it->first].end())
            {
                int sen_counts = gdata.RESULTS_EXT_EXT[it->first][it2->first].size();
                for(int i=0; i<sen_counts; ++i)
                    of << gdata.RESULTS_EXT_EXT[it->first][it2->first][i] << endl;
            }
            of << "#二级扩展句结束" << endl;

        }

        of.flush();
    }

    cout << "[dump_data_stage2]结束..." << endl;

    return;
}

void process_tyccl(string fname, string f_uni, DATA_STRUCT& gdata)
{
    ifstream fin(fname);
    string line;

    string liju_prefix = "#例句:";
    string line_begin = "#------------------------------------";
    string ok_prefix = "[DDv ";
    char delims = '|';


    string CURRENT_W = "";
    string CURRENT_L = "";

    gdata.RESULTS_WS.clear();
    gdata.RESULTS_SENT.clear();
    gdata.RESULTS_AIM.clear();
    gdata.RESULTS_UNI.clear();

    vector<string> dummy_sents;
     while (getline(fin, line))
    {

        trim(line);
        if (line.length() <= 0)
        {
            if(CURRENT_W.length() && CURRENT_L.length() && dummy_sents.size())
            {
                gdata.RESULTS_SENT[CURRENT_W][CURRENT_L] = dummy_sents;
                dummy_sents.clear();
            }
            CURRENT_L = "";
            continue;
        }

        if (line.find(line_begin) != string::npos)
        {
            if(CURRENT_W.length() && CURRENT_L.length() && dummy_sents.size())
            {
                gdata.RESULTS_SENT[CURRENT_W][CURRENT_L] = dummy_sents;
                dummy_sents.clear();
            }
            CURRENT_W = "";
            CURRENT_L = "";
            continue;
        }

        if (line.find(ok_prefix) != string::npos)
        {
            if(CURRENT_W.length() && CURRENT_L.length() && dummy_sents.size())
            {
                gdata.RESULTS_SENT[CURRENT_W][CURRENT_L] = dummy_sents;
                dummy_sents.clear();
            }

            CURRENT_L = "";
            CURRENT_W = line.substr(ok_prefix.length(), line.find(']')-ok_prefix.length());
            continue;
        }

        if ((CURRENT_W.length() != 0) && (line.find(flag_a) == 0) && (line.find(flag_c) != string::npos) )
        {
            if(CURRENT_W.length() && CURRENT_L.length() && dummy_sents.size())
            {
                gdata.RESULTS_SENT[CURRENT_W][CURRENT_L] = dummy_sents;
                dummy_sents.clear();
            }

            CURRENT_L = line.substr(0, line.find(flag_c) + flag_c.length());
            string line_x = line.substr(line.find(flag_c) + flag_c.length());
            trim(line_x);

            vector<string> words_v;
            split(line_x, ' ', words_v);
            set<string> words_s(words_v.begin(), words_v.end());
            gdata.RESULTS_WS[CURRENT_W][CURRENT_L] = words_s;

            for(int i=0; i<words_v.size(); ++i)
            {
                gdata.RESULTS_AIM[words_v[i]].insert(CURRENT_W);
            }

            continue;
        }

        if (line.length()>0 && (CURRENT_W=="" || CURRENT_L==""))
            continue;

        string target = " " + CURRENT_W + "/" + CURRENT_L + " ";

        map<string, vector<string> > dummy_it;
        if (line.find(liju_prefix) != string::npos)
        {
            line = line.substr(liju_prefix.length());

            if(line.find(CURRENT_W) == string::npos)
            {
                continue;
            }
            else
            {
                line.replace(line.find(CURRENT_W), CURRENT_W.length(), target);
                trim(line);
                dummy_sents.push_back(line);
            }
        }
        else if(line.find(flag_d) != string::npos)
        {
            vector<string> sents_v;
            split(line, delims, sents_v);

            for(int i=0; i<sents_v.size(); ++i)
            {
                if(sents_v[i].find(flag_d) != string::npos)
                {
                    sents_v[i].replace(sents_v[i].find(flag_d), flag_d.size(), target);
                    trim(sents_v[i]);
                    dummy_sents.push_back(sents_v[i]);
                }
            }
        }

    }

    fin.close();

    ifstream fin2(f_uni);
    string line_x = "";

    while (getline(fin2, line))
    {
        CURRENT_W = line.substr(0, line.find("|"));
        line_x = line.substr(line.find("|")+1);
        if(string_counts(line_x, flag_c) == 1 &&
            gdata.RESULTS_AIM.find(CURRENT_W) != gdata.RESULTS_AIM.end())
        {
            gdata.RESULTS_UNI.insert(CURRENT_W);
        }
    }

    cout << "统计个数:" << endl;
    cout << "RESULTS_AIM:" << gdata.RESULTS_AIM.size() << endl;
    cout << "RESULTS_UNI:" << gdata.RESULTS_UNI.size() << endl;


    fin2.close();

    return;
}

void add_ext_sents(string str, string c_word, string c_yix, DATA_STRUCT& gdata)
{
    if(gdata.RESULTS_EXT.find(c_word) == gdata.RESULTS_EXT.end())
    {
        map < string, vector< string> > dummy_map;
        gdata.RESULTS_EXT[c_word] = dummy_map;
    }
    if(gdata.RESULTS_EXT[c_word].find(c_yix) == gdata.RESULTS_EXT[c_word].end())
    {
        vector< string> dummy_vector;
        gdata.RESULTS_EXT[c_word][c_yix] = dummy_vector;
    }

    gdata.RESULTS_EXT[c_word][c_yix].push_back(str);

    return;
}

void add_ext_ext_sents(string str, string c_word, string c_yix, DATA_STRUCT& gdata)
{
    if(gdata.RESULTS_EXT_EXT.find(c_word) == gdata.RESULTS_EXT_EXT.end())
    {
        map < string, vector< string> > dummy_map;
        gdata.RESULTS_EXT_EXT[c_word] = dummy_map;
    }
    if(gdata.RESULTS_EXT_EXT[c_word].find(c_yix) == gdata.RESULTS_EXT_EXT[c_word].end())
    {
        vector< string> dummy_vector;
        gdata.RESULTS_EXT_EXT[c_word][c_yix] = dummy_vector;
    }

    gdata.RESULTS_EXT_EXT[c_word][c_yix].push_back(str);

    return;
}

static inline string join_string(const vector<string> words_v, int from=0, int to=0/*include*/)
{
    string str = "";
    if(to == 0)
    {
        to = words_v.size() - 1;
    }

    for(int i=from; i<= to; ++i)
        str += words_v[i];

    return str;
}

void collect_sents_ext_ext(string yl_file, DATA_STRUCT& gdata)
{
    ifstream fin(yl_file);
    string line;
    int line_num = 0;

    gdata.RESULTS_EXT_EXT.clear();

    vector<string> line_words;
    while (getline(fin, line))
    {
        trim(line);
        if(line.length() ==0 || line.length() > 100)
            continue;

        ++ line_num;
        if((line_num % 5000) == 0)
            cout << "C2:" << line_num << endl;

        line_words.clear();
        split(line, ' ', line_words);
        int line_len = line_words.size();
        if(line_len <=3 || line_len>=100)
            continue;

        string line_org = join_string(line_words);

        for(int index=0; index<line_len; ++index)
        {
            if(gdata.RESULTS_RCPTS.find(line_words[index]) != gdata.RESULTS_RCPTS.end())
            {
                for(auto it=gdata.RESULTS_RCPTS[line_words[index]].begin(); 
                        it!=gdata.RESULTS_RCPTS[line_words[index]].end(); ++it)
                {
                    if(line_org.find(it->first) != string::npos)
                    {
                        for(auto it2=it->second.begin(); it2!=it->second.end(); ++it2)
                        {
                            vector<string> dist_v;
                            split(*it2, '/', dist_v);
                            string word_index = dist_v[0];
                            string word_yix = dist_v[1];
                            string target = " " + line_words[index] + "/" + word_yix + " ";

                            vector<string> vect2 = line_words;
                            for(int i=0; i<vect2.size(); ++i)
                                if(vect2[i] == line_words[index])
                                    vect2[i] = target;
                            string dest = join_string(vect2);
                            add_ext_ext_sents(dest, word_index, word_yix, gdata);
                        }
                    }
                }
            }
        }

    }

    return;

}

//分析语料，将简单句归类
void collect_sents_ext(string yl_file, DATA_STRUCT& gdata)
{
    ifstream fin(yl_file);
    string line;
    int line_num = 0;

    gdata.RESULTS_EXT.clear();
    gdata.RESULTS_RCPTS.clear();

    vector<string> line_words;
    while (getline(fin, line))
    {
        trim(line);
        if(line.length() ==0 || line.length() > 100)
            continue;

        ++ line_num;
        if((line_num % 5000) == 0)
            cout << "C1:" << line_num << endl;

        line_words.clear();
        split(line, ' ', line_words);
        int line_len = line_words.size();
        if(line_len <=3 || line_len>=100)
            continue;

        string curr_word = "";
        string index_word = "";
        set<string> ::iterator it_s;
        for(int i=0; i<line_len; ++i)
        {
            curr_word = line_words[i];
            if( gdata.RESULTS_UNI.find(curr_word) != gdata.RESULTS_UNI.end())
            {
                for(auto it1 = gdata.RESULTS_AIM[curr_word].begin(); 
                        it1 != gdata.RESULTS_AIM[curr_word].end(); ++it1)
                {
                    index_word = *it1;  //#同义词索引词
                    // 各个意项
                    for(auto it2 = gdata.RESULTS_WS[index_word].begin();
                        it2 != gdata.RESULTS_WS[index_word].end(); ++it2)
                    {
                        //cout << it2->first <<  endl;
                        if(it2->second.find(curr_word) != it2->second.end())
                        {
                            string target = " " + curr_word + "/" + it2->first + " ";
                            vector<string> vect2 = line_words;  //安全的值拷贝
                            vect2[i] = target;
                            string dest = join_string(vect2);
                            trim(dest);
                            add_ext_sents(dest, index_word, it2->first, gdata);
                            //cout << index_word << ":" << dest << endl;

                            //进行待选扩展扩充
                            vect2.clear();
                            vect2 = line_words;
                            int pre = (i>=2) ? (i-2) : 0;
                            int nex = (i+2<line_len) ? (i+2): (line_len-1);
                            if(nex - pre >= 4)
                            {
                                for(auto it3=gdata.RESULTS_WS[index_word][it2->first].begin();
                                    it3!=gdata.RESULTS_WS[index_word][it2->first].end(); ++it3)
                                {
                                    if(*it3 == curr_word)
                                        continue;

                                    vect2[i] = *it3;
                                    string dest = join_string(vect2, pre, nex);
                                    string disr = index_word + '/' + it2->first;
                                    if (gdata.RESULTS_RCPTS.find(*it3) == gdata.RESULTS_RCPTS.end())
                                    {
                                        map<string, set<string> > dummy_map;
                                        gdata.RESULTS_RCPTS[*it3] = dummy_map;
                                    }
                                    gdata.RESULTS_RCPTS[*it3][dest].insert(disr);
                                }
                            }
                        }
                    }
                }
            }
        }

    }

    cout << "RESULTS_RCPTS:" << gdata.RESULTS_RCPTS.size() << endl;
    return;
}

DATA_STRUCT g_data;
int main(int argc, char* argv[])
{
    string IN_FILE = "结果文件_TAO_T.txt";
    string YL_FILE = "v5语料库_p.txt";
    string UNI_FILE = "同义词分词库.txt";

    string RES_FILE1 = YL_FILE+".out1";
    string RES_FILE2 = YL_FILE+".out2";

    process_tyccl(IN_FILE, UNI_FILE, g_data);
    collect_sents_ext(YL_FILE, g_data);
    dump_data_stage1(RES_FILE1, g_data);

    collect_sents_ext_ext(YL_FILE, g_data);
    dump_data_stage2(RES_FILE2, g_data);

}

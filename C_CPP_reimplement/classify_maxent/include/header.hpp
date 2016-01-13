#ifndef HEADER_H_H
#define HEADER_H_H

#include "jieba.hpp"
#include <vector>
#include <map>
#include <string>

#include <iomanip>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>


using namespace std;

//jieba
P_Jieba jieba_initialize(void);
bool jieba_cut(P_Jieba jieba, string str, vector<std::string> &elems);
void jieba_close(P_Jieba jieba);

//
vector<string> split(const string &s, char delim);
vector<string> &split(const string &s, char delim, vector<std::string> &elems);
bool unique_vector( vector<int> &vec);
static const string dump_file = "dump_cpp.dat_6d932b7aad";

enum MAX_ENT_TYPE { 
    max_ent_gis = 1, 
    max_ent_megam,
};

typedef struct _CLASS_DATA_STRUCT
{
    string data_path;
    P_Jieba jieba;
    
    vector<std::string> train_tags;
    vector<int> sorted_wscores;     //需要保持是有序的

    vector<std::string>   train_w;      //不断的push_back，索引放入下面两个map中
    map<std::string, int> train_w_id;   //互相的快速查找
    map<int, std::string> train_id_w;

    map<int, vector< vector<int> > > train_info;    //训练集
    map<int, vector< vector<int> > > test_info;     //测试集   // HARDCODE 500

    //训练结果
    int BEST_N;
    enum MAX_ENT_TYPE train_type;
    map<int, int> useful_words; //word_id, BEST_N_word_index
    map<int, int> n_ft_index;
    vector<double> n_weight;

} CLASS_DATA_STRUCT, *P_CLASS_DATA_STRUCT;

static const int TAG_SHIFT = 8; // 256足够足够了


bool prep_train_data(CLASS_DATA_STRUCT &cds, string dirname);
static bool process_train_file(CLASS_DATA_STRUCT &cds, int tag_id, 
    map<int, int> &word_fd, map<int, int> &tmp_word_fd);
bool load_train_data(CLASS_DATA_STRUCT &cds, string filename);
bool save_train_data(CLASS_DATA_STRUCT &cds, string filename);
bool train_classifyer_gis(CLASS_DATA_STRUCT &cds, int BEST_N, int iter_count, bool eval_mode);
bool train_classifyer_megam(CLASS_DATA_STRUCT &cds, int BEST_N, bool eval_mode);
static bool eval_classifyer(CLASS_DATA_STRUCT &cds, int BEST_N, double &store);
void eval_classifyers_and_args(CLASS_DATA_STRUCT &cds);
bool predict_it(CLASS_DATA_STRUCT &cds, const vector<std::string> str, 
     map<int, double> & store);

static inline void backtrace_info(int)
{
    int j, nptrs;
#define BT_SIZE 100
    char **strings;
    void *buffer[BT_SIZE];
    
    nptrs = backtrace(buffer, BT_SIZE);
    fprintf(stderr, "backtrace() returned %d addresses\n", nptrs);
    
    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL) 
    {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }

    for (j = 0; j < nptrs; j++)
        fprintf(stderr, "%s\n", strings[j]);

    free(strings);
    
#undef BT_SIZE

    exit(-1);
}

// Global Data
extern P_Jieba jieba;
extern CLASS_DATA_STRUCT cds;
extern bool verbose;

#endif

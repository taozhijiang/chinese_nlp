#ifndef HEADER_H_H
#define HEADER_H_H

#include "jieba.hpp"
#include <vector>
#include <map>
#include<string>

#include <execinfo.h>
#include <signal.h>

using namespace std;

//jieba
P_Jieba jieba_initialize(void);
bool jieba_cut(P_Jieba jieba, string str, vector<std::string> &elems);
void jieba_close(P_Jieba jieba);

//
vector<string> split(const string &s, char delim);
vector<string> &split(const string &s, char delim, vector<std::string> &elems);

typedef struct _CLASS_DATA_STRUCT
{
    vector<std::string> train_tags;
    vector<int> sorted_wscores;     //需要保持是有序的

    map<std::string, int> train_w_id;   //互相的快速查找
    map<int, std::string> train_id_w;

    map<int, vector< vector<int> > > train_info;    //训练集

    map<int, vector< vector<int> > > test_info;     //测试集   // HARDCODE 500

    //训练结果
    map<int, int> useful_words; //word_id, word_index
    map<int,double> priors;
    map<int,vector<double> > multinomial_likelihoods;
    map<int,vector<double> > bernoulli_means;

    map<int,double> cache_bl ;     //做的一个缓存，表示所有词都没出现的伯努利概率（对数）

} CLASS_DATA_STRUCT, *P_CLASS_DATA_STRUCT;

enum CLASSIFIER {
    BernoulliNB = 1,
    MultinomialNB = 2,
};

bool load_train_data(string filename, CLASS_DATA_STRUCT &cds);
bool train_classifyer(CLASS_DATA_STRUCT &cds, int BEST_N, double alpha, bool eval_mode);
static bool eval_classifyer(CLASS_DATA_STRUCT &cds, int BEST_N, map<int, double> &store);
void eval_classifyers_and_args(CLASS_DATA_STRUCT &cds);
bool predict_it(CLASS_DATA_STRUCT &cds, vector<std::string> str, 
    enum CLASSIFIER class_t, map<int, double> & store);

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

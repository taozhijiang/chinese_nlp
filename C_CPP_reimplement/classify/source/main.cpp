#include <iostream>
#include <string>


using namespace std;

#include "header.hpp"
#include <unistd.h>


P_Jieba jieba = NULL;
CLASS_DATA_STRUCT cds;
bool verbose = false;

static void usage(void)
{
    cout << "  **************************************************************" << endl;
    cout << "    USAGE:" <<endl;
    cout << "    classify [ -d datadir] [-b best_n] [-t text] [-v] [-h] [-e]" << endl;
    cout << "      -d datadir 指向训练数据的位置" << endl;
    cout << "      -b best_n  卡方分布选取最优词的数目" << endl;
    cout << "      -t text     需要测试的文本 " << endl;
    cout << "      -v           显示更多处理信息 " << endl;
    cout << "      -e           测试算法各个参数下的性能 " << endl;
    cout << "  *************************************************************" << endl;
}

int main(int argc, char** argv) 
{


    int opt_g = 0;
    int opterr = 0;

    verbose = false;
    int best_n = 5000;
    int eval_mode = false;

    string str_test = "早先来自奥迪内部消息称，奥迪2016年将在工厂和设备领域投资33亿欧元。根据奥迪上年发布的中期发展规划，2015年至2019年，奥迪将总计投资170亿欧元，折合平均每年投资3";

    string data_dir = "../../data_dir/ClassFile_F/";

    while( (opt_g = getopt(argc, argv, "d:t:b:vhe")) != -1 )
    {
        switch(opt_g)
        {
            case 'd':
                data_dir = optarg;
                break;
            case 't':
                str_test = optarg;
                break;
            case 'v':
                verbose = atoi(optarg);
                break;
            case 'b':
                best_n = atoi(optarg);
                break;
            case 'e':
                eval_mode = true;
                break;
            case 'h':
            default:
                usage();
                exit(EXIT_SUCCESS);
        }
    }


    jieba = jieba_initialize();
    cds.jieba = jieba;
    cds.data_path = "";

    if(access(dump_file.c_str(),R_OK) == 0)
    {
        cout << "LOAD FROM PREDUMP DATA!" << endl;
        load_train_data(cds, dump_file);   
    }
    else
    {
        cout << "TRAIN FROM RAW DATA!" << endl;
        prep_train_data(cds, data_dir);
    }

    if(eval_mode)
    {
        cout << "Enter eval mode..." << endl;
        eval_classifyers_and_args(cds);
    }

    cout << cds.data_path << endl;
   
    
    // optional
    //eval_classifyers_and_args(cds);

    // train it last
    train_classifyer(cds, best_n, 1, false);

    vector<std::string> store;
    map<int, double> ret;
    map<int, double > :: iterator it;

    cout << str_test << endl;

    if (jieba_cut(jieba, str_test, store))
    {
        if ( predict_it(cds, store, BernoulliNB, ret) )
        {
            cout << "BernoulliNB:" << endl;
            cout << "Predict Result:" << cds.train_tags[(int)ret[0]] << endl;
            for ( it = ret.begin() ; it != ret.end(); ++it)
            {
                if ( it->first == 0) continue;
                cout << "\t" << cds.train_tags[it->first] << ":" << setiosflags(ios::fixed) << it->second << endl;   
            }
        }


        if ( predict_it(cds, store, MultinomialNB, ret) )
        {
            cout << "MultinomialNB:" << endl;
            cout << "Predict Result:" << cds.train_tags[(int)ret[0]] << endl;
            for ( it = ret.begin() ; it != ret.end(); ++it)
            {
                if ( it->first == 0) continue;
                cout << "\t" << cds.train_tags[it->first] << ":" << setiosflags(ios::fixed) << it->second << endl;   
            }
        }

    }
    
    return 0; 
}

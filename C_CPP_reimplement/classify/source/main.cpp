#include <iostream>
#include <string>


using namespace std;

#include "header.hpp"


P_Jieba jieba = NULL;
CLASS_DATA_STRUCT cds;
bool verbose = false;

int main(int argc, char** argv) 
{

    jieba = jieba_initialize();
    cds.jieba = jieba;
    prep_train_data(cds, "../../data_dir/ClassFile_F/");

    cout << cds.data_path << endl;
   
    
//    load_train_data("dump_cpp.dat_v4_lite", cds);
    
    // optional
    //eval_classifyers_and_args(cds);

    // train it last
    train_classifyer(cds, 5000, 1, false);

    //string str = "有的球迷认为曾诚和金城武长的有些相像，对此，曾诚表示：谢谢球迷们的厚爱，可能某些五官当中的特质有一点";
    string str = "早先来自奥迪内部消息称，奥迪2016年将在工厂和设备领域投资33亿欧元。根据奥迪上年发布的中期发展规划，2015年至2019年，奥迪将总计投资170亿欧元，折合平均每年投资3";
    vector<std::string> store;
    map<int, double> ret;
    map<int, double > :: iterator it;

    cout << str << endl;

    if (jieba_cut(jieba, str, store))
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

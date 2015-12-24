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
    load_train_data("dump_cpp.dat_v4", cds);
    
    // optional
    //eval_classifyers_and_args(cds);

    // train it last
    train_classifyer(cds, 5000, 1, false);

    string str = "　有的球迷认为曾诚和金城武长的有些相像，对此，曾诚表示：谢谢球迷们的厚爱，可能某些五官当中的特质有一点";
    vector<std::string> store;
    map<int, double> ret;
    map<int, double > :: iterator it;

    if (jieba_cut(jieba, str, store))
    {
        #if 0
        predict_it(cds, store, BernoulliNB, ret);
        cout << "BernoulliNB:" << endl;
        for ( it = ret.begin() ; it != ret.end(); ++it)
        {
            cout<< "\t" << cds.train_tags[it->first] << ":" << ret[it->second] << endl;
        }
        #endif

        predict_it(cds, store, MultinomialNB, ret);
        cout << "MultinomialNB:" << endl;
        for ( it = ret.begin() ; it != ret.end(); ++it)
        {
            cout<< "\t" << cds.train_tags[it->first] << ":" << ret[it->second] << endl;
        }
    }

    return 0; 
}

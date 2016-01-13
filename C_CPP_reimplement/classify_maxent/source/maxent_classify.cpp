// Maxium Entropy Classify Implementation


// Modified by taozhijiang@gmail.com

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <map>
#define _USE_MATH_DEFINES	/* for const*/
#include <cmath>
#include <set>
#include <time.h>

#include <float.h>

#include "header.hpp"


using namespace std;


static bool generate_encode(CLASS_DATA_STRUCT &cds, int tag_id, 
        const vector<int>& from, map<int, double>& to)
{
    int tmp_key = 0;
    int corr_count = 0;
    int FT_SIZE = cds.n_ft_index.size();

    to.clear();

    for (int i = 0; i < from.size(); ++i)  //每个词
    {
        if ( cds.useful_words.find(from[i]) != cds.useful_words.end())
        {
            tmp_key = from[i] << TAG_SHIFT | tag_id;
            if(cds.n_ft_index.find(tmp_key) != cds.n_ft_index.end())
            {
                ++ corr_count;
                to[cds.n_ft_index[tmp_key]] = 1;
            }
        }
    }

    if( cds.train_type == max_ent_gis)
    {
        // corr relation
        to[FT_SIZE] = (cds.BEST_N + 1 - corr_count);   
    }
    else if( cds.train_type == max_ent_megam)
    {
        to[FT_SIZE + tag_id - 1] = 1;   // __always_on__
    }
    else
    {
        cerr << "UNKNOWN TRAIN TYPE:" << endl;
        exit(EXIT_FAILURE);
    }

    return true;
}


bool prob_classify(CLASS_DATA_STRUCT &cds, map<int, double>& prob_dict,
    map<int, double>& pdist)
{
    map<int, double> ::iterator it_id;
    pdist.clear();

    if(prob_dict.empty() || prob_dict.size() != (cds.train_tags.size() -1 ))
    {
        cerr << " error prob_dict size!" << endl;
        exit(-1);
    }

    // calc pdist
    it_id = prob_dict.begin();
    double start_val = it_id->second;
    double base;

    ++ it_id;
    //cout << start_val;
    for(/**/; it_id != prob_dict.end(); ++it_id)
    {
        //cout << " " << it_id->second << " ";
        // simple it
        //pdist[it_id->first] = pow(2,log(1.0/prob_dict.size()));
        base = start_val < it_id->second? start_val : it_id->second;
        start_val = base + log2(pow(2,(start_val-base)) + pow(2,(it_id->second-base)));
    }

    for(it_id = prob_dict.begin(); it_id != prob_dict.end(); ++it_id)
    {
        pdist[it_id->first] = pow(2, (prob_dict[it_id->first] - start_val));
    }

    return true;
}

bool train_classifyer_megam(CLASS_DATA_STRUCT &cds, int best_n, bool eval_mode)
{
    // For segment fault debug
    struct sigaction sa;
    sa.sa_handler = backtrace_info;
    sigaction(SIGSEGV, &sa, NULL); 

    cout << "BEGIN TO TRAIN MEGAM ..." << endl;
    // 选取指定的测试信息
    cds.useful_words.clear();
    if (best_n > cds.sorted_wscores.size())
        best_n = cds.sorted_wscores.size();
    // 重新复制有效词的索引
    for(int i = 0; i < best_n; i++)
        cds.useful_words[cds.sorted_wscores[i]] = i;

    cds.BEST_N = best_n;
    cds.n_ft_index.clear();

    char* TMP_INPUT_FILE = "nltk-megam.tmp";

    //将测试数据集添加到训练数据集后面
    if( ! eval_mode)
    {
        cout << "MERGE TEST TO TRAIN!!!" << endl;
        for(int i = 1; i<cds.train_tags.size(); ++i)
        {
            for(int j = 0; j < cds.test_info.size(); ++j)
                cds.train_info[i].push_back(cds.test_info[i][j]);
        }
    }

    /**
     * calc empirical_count
     */

     // STAGE 1 fname_fval_lable id map
    map<int, vector< vector<int> > > :: iterator it;
    cds.n_ft_index.clear();
    for (it = cds.train_info.begin(); it != cds.train_info.end(); ++ it)
    {
        int tag_id = it->first;

        vector< vector<int> > t_items = it->second;  //训练文档
        for ( int i = 0; i< t_items.size(); ++i )
        {            
            vector<int> t_item = t_items.at(i);     //文档中的词
            unique_vector(t_item);
            int tmp_key = 0;
            int tmp_val = 0;
            for (int j = 0; j < t_item.size(); j++)  //每个词
            {
                // 看该单词是不是在选定的特征词当中
                if ( cds.useful_words.find(t_item[j]) != cds.useful_words.end())
                {
                    tmp_key = t_item[j] << TAG_SHIFT | tag_id;
                    //如果尚未添加到feature的数组中，添加key->index的mapping
                    if(cds.n_ft_index.find(tmp_key) == cds.n_ft_index.end())
                    {
                        tmp_val = cds.n_ft_index.size();
                        cds.n_ft_index[tmp_key] = tmp_val;
                        // Critical !!!!
                        //cds.n_ft_index[tmp_key] = cds.n_ft_index.size();   
                    }
                }
            }
        }
    }

    map<int, double> ::iterator it_id;
    // Write a training file for megam.
    ofstream fout(TMP_INPUT_FILE);

    // 计算estimated_fcount
    for (it = cds.train_info.begin(); it != cds.train_info.end(); ++it)
    {
        int tag_id = it->first;
        map<int, double> en_features;

        vector< vector<int> > t_items = it->second;  //训练文档
        for ( int i = 0; i< t_items.size(); i++ )
        {
            vector<int> t_item = t_items.at(i);     //文档中的词
            unique_vector(t_item);

            //???
            if(t_item.empty())
                continue;

            fout << (tag_id -1);    // ATTENTION: SHOULB BEGIN WITH 0!!!

            for( int pre_tid = 1; pre_tid < cds.train_tags.size(); ++ pre_tid)
            {
                generate_encode(cds, pre_tid, t_item, en_features);
                fout << " #";
                for(it_id = en_features.begin(); it_id != en_features.end(); ++it_id)
                    fout <<" " << it_id->first;
            }
            fout << endl;
        }
    }
    fout.close();

    //system(" wc nltk-megam.tmp ");

    int pipefd[2];
    pipe(pipefd);

    map<int, double> weights;
    if (fork() == 0)
    {
        close(pipefd[0]);    // close reading end in the child
        dup2(pipefd[1], STDOUT_FILENO);  // send stdout to the pipe
        dup2(pipefd[1], STDERR_FILENO);  // send stderr to the pipe
        close(pipefd[1]);    // this descriptor is no longer needed

        char *const exec_args[] = 
        {
            "./megam_i686.opt",
            "-quiet",
            "-nobias",
            "-repeat", "10",
            "-explicit", 
            "-lambda", "0.00", 
            "-tune", 
            "multiclass", 
            TMP_INPUT_FILE, 
            (char  *) NULL,
        };

        execvp("./megam_i686.opt", exec_args);
        remove(TMP_INPUT_FILE);
        exit(0);
    }
    else
    {
        // parent
        char buffer[1024];
        close(pipefd[1]);  // close the write end of the pipe in the parent
        FILE* fp = fdopen(pipefd[0],"r");

        while (fgets(buffer, sizeof(buffer)-1, fp) != 0)
        {
            vector<string> tokens = split(buffer,' ');
            if(tokens.size() != 2)
            {
                cout << ">>>:" << buffer;
                continue;
            }
            weights[atoi(tokens[0].c_str())] = atof(tokens[1].c_str());
        }
    }

    if(weights.size() != (cds.n_ft_index.size() + cds.train_tags.size() -1 )) // __always_on__
    {
        cout << "MISMATCH size:" << weights.size() << "-" << cds.n_ft_index.size() << endl;
        return false;
    }

    cds.n_weight.clear();
    for(int i=0; i<weights.size(); ++i)
        cds.n_weight.push_back( weights[i]*log2(M_E) ); //ln->log2

    cout << "MEGAM FINISHED with size:" << cds.n_weight.size() << endl;

    if(!eval_mode)
    {

        if(!cds.data_path.length())
        {
            cout << "LOADED TYPE, DO NOT STORE TO FILE!" << endl;
        }
        else
        {
            cout << "SAVE DATA TO FILE!" << endl;
            save_train_data(cds, dump_file);   
        }

        cout << "TRAIN DONE, RELEASE TRAIN DATA!" << endl;
        cds.train_info.clear();   
        cds.test_info.clear();
    }

    return true;
}

// 同时训练出伯努利分布和多项式分布的朴素贝叶斯分类器
// 如果 eval_mode == true  测试模式
// 如果 eval_mode == false 运行模式，此时test_info也会并入训练数据集合，同时释放不用空间
//                                      应为数据会被清除，所以这种模式只能运行一次
bool train_classifyer_gis(CLASS_DATA_STRUCT &cds, int best_n, int iter_count, bool eval_mode)
{

    // For segment fault debug
    struct sigaction sa;
    sa.sa_handler = backtrace_info;
    sigaction(SIGSEGV, &sa, NULL); 

    cout << "BEGIN TO TRAIN GIS ..." << endl;
    // 选取指定的测试信息
    cds.useful_words.clear();
    if (best_n > cds.sorted_wscores.size())
        best_n = cds.sorted_wscores.size();
    // 重新复制有效词的索引
    for(int i = 0; i < best_n; i++)
        cds.useful_words[cds.sorted_wscores[i]] = i;

    cds.BEST_N = best_n;
    cds.n_ft_index.clear();
    cds.n_weight.clear();

    unsigned int FT_SIZE = 0;

        //将测试数据集添加到训练数据集后面
    if( ! eval_mode)
    {
        cout << "MERGE TEST TO TRAIN!!!" << endl;
        for(int i = 1; i<cds.train_tags.size(); ++i)
        {
            for(int j = 0; j < cds.test_info.size(); ++j)
                cds.train_info[i].push_back(cds.test_info[i][j]);
        }
    }

    /**
     * calc empirical_count
     *
     * Support the need of correction feature in GIS.
     */

     // STAGE 1 fname_fval_lable id map
    map<int, vector< vector<int> > > :: iterator it;
    for (it = cds.train_info.begin(); it != cds.train_info.end(); ++ it)
    {
        int tag_id = it->first;

        vector< vector<int> > t_items = it->second;  //训练文档
        for ( int i = 0; i< t_items.size(); ++i )
        {            
            vector<int> t_item = t_items.at(i);     //文档中的词
            unique_vector(t_item);
            int tmp_key = 0;
            int tmp_val = 0;
            for (int j = 0; j < t_item.size(); j++)  //每个词
            {
                // 看该单词是不是在选定的特征词当中
                if ( cds.useful_words.find(t_item[j]) != cds.useful_words.end())
                {
                    tmp_key = t_item[j] << TAG_SHIFT | tag_id;
                    //如果尚未添加到feature的数组中，添加key->index的mapping
                    if(cds.n_ft_index.find(tmp_key) == cds.n_ft_index.end())
                    {
                        tmp_val = cds.n_ft_index.size();
                        cds.n_ft_index[tmp_key] = tmp_val;
                        // Critical !!!!
                        //cds.n_ft_index[tmp_key] = cds.n_ft_index.size(); 
                    }
                }
            }
        }
    }

    double Cinv = 1.0f/(cds.BEST_N+1);
    FT_SIZE      = cds.n_ft_index.size();
    double weight        [FT_SIZE + 1]  = {0.0f,};
    double last_weight  [FT_SIZE + 1]  = {0.0f,};

    cout << "特征维度：" << best_n << "/"<< FT_SIZE << endl;

    map<int, int> ::iterator it_ii;
    map<int, double> ::iterator it_id;

    // STAGE 2, calculate empirical_fcount
    map<int, double> empirical_fcount;
    map<int, double> log_empirical_fcount;

    for (it = cds.train_info.begin(); it != cds.train_info.end(); it++)
    {
        int tag_id = it->first;

        vector< vector<int> > t_items = it->second;  //训练文档
        for ( int i = 0; i< t_items.size(); ++i )
        {
            vector<int> t_item = t_items.at(i);     //文档中的词
            unique_vector(t_item);
            map<int, double> en_features;

            generate_encode(cds, tag_id, t_item, en_features);
            for(it_id = en_features.begin(); it_id != en_features.end(); ++it_id)
            {
                empirical_fcount[it_id->first] += it_id->second;
            }
        }
    }

    for(it_id = empirical_fcount.begin(); it_id != empirical_fcount.end(); ++it_id)
    {
        log_empirical_fcount[it_id->first] = log(it_id->second);
        //cout << empirical_fcount[it_id->first] << "/" << log_empirical_fcount[it_id->first] <<"  ";
    }
    empirical_fcount.clear();

    map<int, double> estimated_fcount;
    map<int, double> log_estimated_fcount;
    cout << "迭代次数：";
    for(int iter_round = 0; iter_round<iter_count; ++iter_round)
    {
        cout << iter_round << " ";
        cout.flush();
        for(int i = 0; i< FT_SIZE; ++i) //保存先前的权重
            last_weight[i] = weight[i];
        
        estimated_fcount.clear();
        log_estimated_fcount.clear();

        // 计算estimated_fcount
        for (it = cds.train_info.begin(); it != cds.train_info.end(); ++it)
        {
            int tag_id = it->first;
            map<int, double> en_features;

            vector< vector<int> > t_items = it->second;  //训练文档
            for ( int i = 0; i< t_items.size(); i++ )
            {
                vector<int> t_item = t_items.at(i);     //文档中的词
                unique_vector(t_item);

                //???
                if(t_item.empty())
                    continue;

                // TTT-1  calc prob_dict & pdist
                double total = 0.0f;
                map<int, double> prob_dict;
                map<int, double> pdist;

                for( int pre_tid = 1; pre_tid < cds.train_tags.size(); ++ pre_tid)
                {
                    generate_encode(cds, pre_tid, t_item, en_features);
                    
                    total = 0.0f;
                    for(it_id = en_features.begin(); it_id != en_features.end(); ++it_id)
                    {
                        total += weight[it_id->first] * it_id->second;
                    }
                    prob_dict[pre_tid] = total;
                }

                prob_classify(cds, prob_dict, pdist);
                
                // TTT-2 calc fcount
                // 更新 estimate_fcount
                double prob = 0.0f;
                for( int pre_tid = 1; pre_tid < cds.train_tags.size(); ++ pre_tid)
                {
                    prob = pdist[pre_tid];

                    generate_encode(cds, pre_tid, t_item, en_features);
                    for(it_id = en_features.begin(); it_id != en_features.end(); ++it_id)
                    {
                        estimated_fcount[it_id->first] += prob* it_id->second;
                    }
                }

            }
        }

        // 对数化
        for(it_id = estimated_fcount.begin(); it_id != estimated_fcount.end(); ++it_id)
            log_estimated_fcount[it_id->first] = log(it_id->second);
        
        // 更新权重
        for(it_id = log_empirical_fcount.begin(); it_id != log_empirical_fcount.end(); ++it_id)
        {
            weight[it_id->first] += 
                (it_id->second - log_estimated_fcount[it_id->first]) * Cinv;
        }
    }

    cout << endl;
    
    //cout << "保存权重！" << endl;
    for(int i=0; i<FT_SIZE +1; ++i)
        cds.n_weight.push_back(weight[i]);


    if(!eval_mode)
    {

        if(!cds.data_path.length())
        {
            cout << "LOADED TYPE, DO NOT STORE TO FILE!" << endl;
        }
        else
        {
            cout << "SAVE DATA TO FILE!" << endl;
            save_train_data(cds, dump_file);   
        }

        cout << "TRAIN DONE, RELEASE TRAIN DATA!" << endl;
        cds.train_info.clear();   
        cds.test_info.clear();
    }

    return true;
}

void eval_classifyers_and_args(CLASS_DATA_STRUCT &cds)
{
    
    cout << "BEGIN TO TEST CLASSIFIER!\n"<< endl;

    int best_ns[] = {2000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, 25000, 30000, 35000};

    double ret;

    for (int i=0; i< sizeof(best_ns)/sizeof(best_ns[0]); i++)
    {
        time_t t1 = time(NULL);
        eval_classifyer(cds, best_ns[i], ret);
        fprintf(stdout, "BEST_N:%d\tSCORE:%f\tTIME:%ldsecs\n", best_ns[i], ret,
            time(NULL) - t1);
    }
}

// 测试分类结果
static bool eval_classifyer(CLASS_DATA_STRUCT &cds, int best_n, double& store)
{
     // train first
    if (cds.train_type == max_ent_gis)
    {
        train_classifyer_gis(cds, best_n, 40, true);   
    }
    else if(cds.train_type == max_ent_megam)
    {
        train_classifyer_megam(cds, best_n, true);   
    }
    else
    {
        cerr << "UNKNOW TRAIN TYPE:" << cds.train_type << endl;
        exit(-1);
    }

    //遍历 test_info
    map<int, vector< vector<int> > > :: iterator it;

    int total_test = 0;
    int corr_test = 0;

    for (it = cds.test_info.begin(); it != cds.test_info.end(); ++it)   //每一个标签
    {
        int tag_id = it->first; 
        vector< vector<int> > t_items = it->second; //labled docs

        if (verbose)
            cout << "TESTING: " << cds.train_tags[tag_id] << endl;

        for(int it_ds = 0; it_ds < t_items.size(); it_ds++)             //每一个测试文档
        {
            vector<int> t_item = t_items.at(it_ds);     

            // 针对每一个单元进行测试
            unique_vector(t_item);

            double total = 0.0f;
            map<int, double> prob_dict;
            map<int, double> pdist;
            map<int, double> en_features;
            map<int, double> ::iterator it_id;
            double max_prob = 0.0;
            int     max_tag_id = -1;

            for( int pre_tid = 1; pre_tid < cds.train_tags.size(); ++ pre_tid)
            {
                generate_encode(cds, pre_tid, t_item, en_features);
                
                total = 0.0f;
                for(it_id = en_features.begin(); it_id != en_features.end(); ++it_id)
                {
                    total += cds.n_weight[it_id->first] * it_id->second;
                }
                prob_dict[pre_tid] = total;
            }

            prob_classify(cds, prob_dict, pdist);
            
            // TTT-2 calc fcount
            // 更新 estimate_fcount
            for(it_id = pdist.begin(); it_id != pdist.end(); ++it_id)
            {
                //cout << " " << cds.train_tags[it_id->first] << ":" << it_id->second << endl;
                if(it_id->second > max_prob)
                {
                    max_tag_id = it_id->first;
                    max_prob = it_id->second;
                }
            }

            if(max_tag_id == tag_id)
                corr_test += 1;
            total_test += 1;

        }
    }

    store = (double)corr_test / total_test;
    cout << "D:" << corr_test << "/" << total_test << "[" << store << "]" << endl;

    return true;

}


/**
 * 返回值当中，0表示的是概率最大的标签，其余是按照tag_id罗列起来的
 */
bool predict_it(CLASS_DATA_STRUCT &cds, const vector<std::string> store, 
    map<int, double> & ret)
{
    double max_prob = 0.0;
    int     max_tag_id = -1;
    vector<int> word_id;

    for(int i = 1; i < store.size(); ++i)
    {
        auto word_index = cds.train_w_id.find(store[i]);
        if ( word_index != cds.train_w_id.end())
            word_id.push_back(word_index->second);
    }

    // 针对每一个单元进行测试
    unique_vector(word_id);
    if(word_id.empty())
    {
        cout << "NULL PREDICTABLE..." << endl;
        return false;
    }

    double total = 0.0f;
    map<int, double> prob_dict;
    map<int, double> pdist;
    map<int, double> en_features;
    map<int, double> ::iterator it_id;

    for( int pre_tid = 1; pre_tid < cds.train_tags.size(); ++ pre_tid)
    {
        generate_encode(cds, pre_tid, word_id, en_features);

        total = 0.0f;
        for(it_id = en_features.begin(); it_id != en_features.end(); ++it_id)
        {
            total += cds.n_weight[it_id->first] * it_id->second;
        }
        prob_dict[pre_tid] = total;
    }

    ret.clear();
    prob_classify(cds, prob_dict, ret);
    
    // TTT-2 calc fcount
    // 更新 estimate_fcount
    for(it_id = ret.begin(); it_id != ret.end(); ++it_id)
    {
        //cout << " " << cds.train_tags[it_id->first] << ":" << it_id->second << endl;
        if(it_id->second > max_prob)
        {
            max_tag_id = it_id->first;
            max_prob = it_id->second;
        }
    }

    ret[0] = max_tag_id;

    return true;
    
}


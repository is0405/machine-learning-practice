#include "iostream"
#include "learning.h"
#include "cmath"
#include <fstream>
#include <sstream>

using namespace std;

void
learn::machine_learn(int hidden_layer, int hidden, int epoch, long double a, int batch_size, string loss_func_name, string optimizer_func_name, vector<string> active_func_name)
{
    input_num = all_input_data[0].size();
    hidden_layer_num = hidden_layer;
    hidden_num = hidden;
    output_num = all_answer_data[0].size();
    alpha = a;
    loss_name = loss_func_name;
    optimizer_name = optimizer_func_name;
    active_name = active_func_name;
    t = 1;

    if(active_name.size() != hidden_layer_num+1)
    {
        cout << active_name.size() << " " << hidden_layer_num+1 <<" do not match active_name.size()" << endl;
        return;
    }

    init_weight();
    shuffle();
    loss_val = 1e10;

    all_result_data = all_answer_data;

    long double decay = 0.1;
    
    for(int i = 0; i < epoch; ++i)
    {
        for(int j = 0; j < all_answer_data.size(); ++j)
        {   
            input_data = all_input_data[j];
            answer_data = all_answer_data[j];

            forward();
            back_propagation();
            
            all_result_data[j] = result;
        }
        ++t;
        cout << i << " loss: " << loss_val << " accuracy: " << accuracy() << "%" << endl;
        if(i % 5 == 0)
            alpha *= decay;
    }
    
}

bool
learn::read(string file_name, bool input)
{
    const char *FileName = file_name.c_str();
    {
        ifstream ifs(FileName);
        string str;
        if(!ifs)
        {
            cout << file_name <<" ファイルが開けませんでした。" << endl;
            return false;
        }

        while(getline(ifs, str))
        {
            stringstream ss(str);
            string item;
            while(getline(ss, item, ' '))
                if(input)
                    input_data.push_back(stod(item));
                else
                    answer_data.push_back(stod(item));
                
            if(input)
            {
                all_input_data.push_back(input_data);
                input_data.clear();
            }
            else
            {
                all_answer_data.push_back(answer_data);
                answer_data.clear();
            }
        }
    }
    
    return true;
}

bool
learn::write()
{
    string file_name[2] = {"weights.txt", "bias.txt"};
    {
        const char *FileName1 = file_name[0].c_str();
        const char *FileName2 = file_name[1].c_str();
        {
            ofstream ofs1(FileName1);
            if (!ofs1)
            {
                cout << "ファイル1が開けませんでした。" << endl;
                return false;
            }

            ofstream ofs2(FileName2);
            if (!ofs2)
            {
                cout << "ファイル2が開けませんでした。" << endl;
                return false;
            }

            for(auto &a:bias)
            {
                string s2 = "";
                for(auto &t:a)
                    s2 += to_string(t) + "\n";
                ofs2 << s2;
            }

            for(auto &a:weights)
            {
                string s1 = "";
                for(auto &t:a)
                    for(auto &w:t)
                        s1 += to_string(w) + "\n";
                ofs1 << s1;
            }
        }
    }
    return true;
}

void
learn::forward()
{
    predict();
    long double loss_v = loss();
    if(loss_v < loss_val)
    {
        loss_val = loss_v;
        write();
    }
}

void
learn::predict()
{
    vector<long double> data = input_data;
    for(int i = 0; i < hidden_layer_num; ++i)
    {
        // for(int j = 0; j < data.size(); ++j)
        //     cout << i-1 << " " << data[j] << endl;
        hidden_layer[i] = mat_calc(data, i);
        
        for(int j = 0; j < hidden_layer[i].size(); ++j)
        {
            // cout << "s " << i << " " << j << " "<< hidden_layer[i][j] << endl;
            hidden_layer[i][j] = activation(hidden_layer[i][j], active_name[i], true, i);
            // cout << "f " << i << " " << j << " "<< hidden_layer[i][j] << endl;
        }
        
        data = hidden_layer[i];
    }

    // for(int j = 0; j < data.size(); ++j)
    //     cout << hidden_layer_num-1 << " " << data[j] << endl;
    result = mat_calc(data, hidden_layer_num);
    
    for(int i = 0; i < result.size(); ++i)
    {
        
        // cout << "s "<<result[i] << endl;
        result[i] = activation(result[i], active_name[active_name.size()-1], true, hidden_layer_num);
        // cout << "f "<<result[i] << endl;
    }

    
    // for(int j = 0; j < result.size(); ++j)
    //         cout << hidden_layer_num << " " << result[j] << endl;
}

vector<long double>
learn::mat_calc(vector<long double> &data, int num)
{
    int layer_num = num == hidden_layer_num ? output_num : hidden_num;
    vector<long double> ans(layer_num, 0); 
    for(int i = 0; i < layer_num; i++)
    {
        for(int j = 0; j < data.size(); j++)
        {
            ans[i] += data[j] * weights[num][j][i];
        }
        ans[i] += bias[num][i];
    }

    return ans;
}

void
learn::back_propagation()
{
    d_init();
    differential();

    optimizer();

    for(int i = 0; i < weights.size(); ++i)
        for(int j = 0; j < weights[i].size(); ++j)
            for(int k = 0; k < weights[i][j].size(); ++k)
                weights[i][j][k] += d_weights[i][j][k];
            
    for(int i = 0; i < bias.size(); ++i)
        for(int j = 0; j < bias[i].size(); ++j)
            bias[i][j] += d_bias[i][j];
 
    // for(int i = 0; i < d_hidden_layer.size(); ++i)
    //     for(int j = 0; j < d_hidden_layer[i].size(); ++j)
    //         cout <<i <<" " <<  j <<" "<< d_hidden_layer[i][j] << endl;
}

void
learn::d_init()
{
    d_hidden_layer = hidden_layer;
    for(int i = 0; i < hidden_layer.size(); ++i)
        for(int j = 0; j < hidden_layer[i].size(); ++j)
            d_hidden_layer[i][j] = 0;
}

void
learn::differential()
{
    for(int i = hidden_layer_num; -1 < i; --i)
    {
        if(i == hidden_layer_num)
        {
            for(int j = 0; j < output_num; ++j)
            {
                d_bias[i][j] = d_loss(result[j], answer_data[j]) * activation(result[j], active_name[i], false, j);
                for(int k = 0; k < hidden_layer[i-1].size(); ++k)
                {
                    d_weights[i][k][j] = d_bias[i][j] * activation(hidden_layer[i-1][k], active_name[i], false, j);
                    d_hidden_layer[i-1][k] += d_bias[i][j] * weights[i][k][j];
                    // cout <<i <<" " <<  j <<" " << k << " "<< d_bias[i][j] << " " << d_weights[i][k][j] << endl;
                }
            }
        }
        else
        {
            for(int j = 0; j < hidden_layer[i].size(); ++j)
            {
                d_bias[i][j] = activation(hidden_layer[i][j], active_name[i], false, j) * d_hidden_layer[i][j];

                if(i != 0)
                {
                    for(int k = 0; k < hidden_layer[i-1].size(); ++k)
                    {
                        d_weights[i][k][j] = d_bias[i][j] * activation(hidden_layer[i-1][k], active_name[i], false, j);
                        d_hidden_layer[i-1][k] += d_bias[i][j] * weights[i][k][j];
                    }
                }
                else
                {
                    for(int k = 0; k < input_num; ++k)
                    {
                        d_weights[i][k][j] = d_bias[i][j] * input_data[k];
                    }
                }
            }
        }
    }
}

void
learn::optimizer()
{
    const long double beta_1 = 0.9;
    const long double beta_2 = 0.999;
    const long double epsilon = 1e-8;

    for(int i = 0; i < d_weights.size(); ++i)
        for(int j = 0; j < d_weights[i].size(); ++j)
            for(int k = 0; k < d_weights[i][j].size(); ++k)
            {
                if(optimizer_name == "SGD")
                {
                    d_weights[i][j][k] *= -alpha;
                }
                else if(optimizer_name == "Adam")
                {      
                    m[i][j][k] = beta_1 * m[i][j][k] + (1 - beta_1) * d_weights[i][j][k];
                    v[i][j][k] = beta_2 * v[i][j][k] + (1 - beta_2) * pow(d_weights[i][j][k], 2);
                    long double h_m = m[i][j][k] / (1 - pow(beta_1, t));
                    long double h_v = v[i][j][k] / (1 - pow(beta_2, t));
                    d_weights[i][j][k] = - (alpha * h_m) / sqrt(h_v + epsilon);
                    
                }
            }

    
    for(int i = 0; i < d_bias.size(); ++i)
        for(int j = 0; j < d_bias[i].size(); ++j)
        {
            if(optimizer_name == "SGD")
            {
                d_bias[i][j] *= -alpha;
            }
            else if(optimizer_name == "Adam")
            {      
                b_m[i][j] = beta_1 * b_m[i][j] + (1 - beta_1) * d_bias[i][j];
                b_v[i][j] = beta_2 * b_v[i][j] + (1 - beta_2) * pow(d_bias[i][j], 2);
                long double h_m = b_m[i][j] / (1 - pow(beta_1, t));
                long double h_v = b_v[i][j] / (1 - pow(beta_2, t));
                d_bias[i][j] = - (alpha * h_m) / sqrt(h_v + epsilon);
                    
            }
        }
}

void
learn::init_weight()
{
    
    for(int i = 0; i < hidden_layer_num+1; ++i)
    {
        if(i == 0)
        {
            vector<vector<long double>> w(input_num, vector<long double>(hidden_num, 1));
            vector<long double> b(hidden_num, 1);
            
            weights.push_back(w);
            bias.push_back(b);
            hidden_layer.push_back(b);
        }
        else if(i == hidden_layer_num)
        {
            vector<vector<long double>> w(hidden_num, vector<long double>(output_num, 1));
            vector<long double> b(output_num, 1);
            
            weights.push_back(w);
            bias.push_back(b);
            result = b;
        }
        else
        {
            vector<vector<long double>> w(hidden_num, vector<long double>(hidden_num, 1));
            vector<long double> b(hidden_num, 1);
            
            weights.push_back(w);
            bias.push_back(b);
            hidden_layer.push_back(b);
        }
    }


    d_weights = weights;
    d_bias = bias;

    m = weights;
    v = weights;    
    b_m = bias;
    b_v = bias;
}

void
learn::shuffle()
{
    for(int i = 0; i < weights.size(); ++i)
        for(int j = 0; j < weights[i].size(); ++j)
            for(int k = 0; k < weights[i][j].size(); ++k)
            {
                double a = rand() % 100;
                weights[i][j][k] = a / 1000;
            }
            
    for(int i = 0; i < bias.size(); ++i)
        for(int j = 0; j < bias[i].size(); ++j)
        {
            double a = rand() % 10;
            bias[i][j] = a;
        }
}

long double
learn::sigmoid(long double x)
{
    return 1/(1+exp(-x));
}

long double
learn::d_sigmoid(long double x)
{
    return (1 - x) * x;
}

long double
learn::softmax(long double x, int layer_number)
{ 
    long double total = 0;

    if(layer_number == hidden_layer_num)
    {
        for(int i = 0; i < result.size(); i++)
        {	
            total += exp(result[i]);
        }
    }
    else
    {
        for(int i = 0; i < hidden_layer[layer_number].size(); i++)
        {	
            total += exp(hidden_layer[layer_number][i]);
        }
    }

    return exp(x) / total;
}

long double
learn::d_softmax(long double x)
{
    return x * (1 - x);
}

long double
learn::soft_sign(long double x)
{
    return x / (1 + abs(x));
}

long double
learn::d_soft_sign(long double x)
{
    return 1 / pow(1 + abs(x), 2);
}

long double
learn::identify(long double x)
{
    return x;
}

long double
learn::d_identify(long double x)
{
    return 1;
}

long double
learn::activation(long double x, string function_name, bool direction, int number)
{
    if( function_name == "sigmoid")
    {
        if( direction )
            return  sigmoid(x);
        else
            return d_sigmoid(x);
    }
    else if( function_name == "softmax")
    {
        if( direction )
            return softmax(x, number);
        else
            return  d_softmax(x);
    }
    else if(function_name == "softsign")
    {
        if( direction )
            return soft_sign(x);
        else
            return  d_soft_sign(x);
    }
    else if(function_name == "identify")
    {
        if( direction )
            return identify(x);
        else
            return  d_identify(x);
    }
    
    return 0;
}

long double
learn::loss()
{
    if(loss_name == "mean_square")
    {
        return mean_squared_error();
    }
    if(loss_name == "cross_entropy")
    {
        return cross_entropy_error();
    }
  
    return 0;
}

long double
learn::d_loss(long double x, long double y)
{
    if(loss_name == "mean_square")
    {
        return d_mean_squared_error(x, y);
    }
    if(loss_name == "cross_entropy")
    {
        return d_cross_entropy_error(x, y);
    }
  
    return 0;
}

long double
learn::mean_squared_error()
{
    long double loss = 0;
    for(int i = 0; i < answer_data.size(); ++i)
    {
        loss += pow(result[i]-answer_data[i], 2);
    }

    return loss/2;
}

long double
learn::cross_entropy_error()
{
    long double loss = 0;
    long double delta = 1e-7;
    for(int i = 0; i < answer_data.size(); ++i)
    {
        loss += answer_data[i] * log(result[i]+delta);
    }

    return -loss;
}

long double
learn::d_mean_squared_error(long double x, long double y)
{
    return (x - y);
}

long double
learn::d_cross_entropy_error(long double x, long double y)
{
    return -(y/x) + (1-y)/(1-x);
}

long double
learn::accuracy()
{
    long double sum = 0;
    for(int i = 0; i < all_answer_data.size(); ++i)
    {
        for(int j = 0; j < all_answer_data[i].size(); ++j)
        {
            if(int(all_result_data[i][j]) == all_answer_data[i][j])
                ++sum;
        }
    }
    return sum / (long double) all_answer_data.size() * 100;
}

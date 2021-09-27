#include <vector>
#include <iostream>
#include <string>

#ifndef LEARNING_H
#define LEARNING_H

class learn
{
 private:
     std::vector<std::vector<std::vector<long double>>> weights;//重みを保存する
    std::vector<std::vector<long double>> bias;//バイアスを保存する
    std::vector<std::vector<std::vector<long double>>> d_weights;//各重みの微分後の値を保存する
    std::vector<std::vector<long double>> d_bias;//各バイアスの微分後の値を保存する

    std::vector<std::vector<long double>> hidden_layer;//各隠れ層の出力値を保存する
    std::vector<std::vector<long double>> d_hidden_layer;//各隠れ層の出力値を保存する

    std::vector<long double> result;//ニューラルネットワークの計算の結果を保存する

    std::vector<std::string> active_name;//活性化関数を保存する

    std::vector<long double> input_data;//入力層の値を保存する
    std::vector<long double> answer_data;

    std::vector<std::vector<long double>> all_input_data;//入力層の値を保存する
    std::vector<std::vector<long double>> all_answer_data;

    //Adamパラメータ
    std::vector<std::vector<std::vector<long double>>> m;
    std::vector<std::vector<std::vector<long double>>> v;
    std::vector<std::vector<long double>> b_m;
    std::vector<std::vector<long double>> b_v;
    
    int input_num;//入力のニューロン数を保存
    int hidden_layer_num;//隠れ層の数を保存
    int hidden_num;//隠れ層のニューロン数を保存
    int output_num;//出力のニューロン数を保存
    std::string loss_name;
    std::string optimizer_name;
    int t;
    long double loss_val;
    long double alpha;
    long double sigmoid(long double x);
    long double softmax(long double x ,int number);
    long double soft_sign(long double x);
    long double identify(long double x);
    long double d_sigmoid(long double x);
    long double d_softmax(long double x);
    long double d_soft_sign(long double x);
    long double d_identify(long double x);

    long double mean_squared_error();
    long double cross_entropy_error();
    long double d_mean_squared_error(long double x, long double y);
    long double d_cross_entropy_error(long double x, long double y);
    void predict();//順伝播
    void forward();
    void back_propagation();
    void d_init();
    void shuffle();
    std::vector<long double> mat_calc(std::vector<long double> &data, int num);
    long double activation(long double x, std::string function_name, bool direction, int number);
    long double loss();
    long double d_loss(long double x, long double y);
    void init_weight();
    void differential();
    void optimizer();
    
 public:
    void machine_learn(int hidden_layer, int hidden, int epoch, long double a, int batch_size, std::string loss_func_name, std::string optimizer_func_name, std::vector<std::string> active_func_name);
    bool read(std::string file_name, bool input);
    bool write();
};
#endif

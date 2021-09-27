#include "iostream"
#include "learning.h"

using namespace std;
int main()
{
    learn a;
    if(!a.read("teacher2.txt", true))
        cout << "no1" << endl;
    if(!a.read("answer2.txt", false))
        cout << "no2" << endl;

    int hidden_layer_num = 1;
    vector<string> active(hidden_layer_num+1, "identify");
    a.machine_learn(hidden_layer_num, 10, 100, 0.1, 100, "mean_square", "Adam", active);

    return 0;
}

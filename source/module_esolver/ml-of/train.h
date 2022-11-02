#include <torch/torch.h>
#include "./nn_of.h"
#include "/home/dell/2_software/libnpy/libnpy/include/npy.hpp"

class Train{
public:
    Train(){};
    
    NN_OF nn;

    void setPara(int nrxx, int ninpt, int nbatch);

    void loadData();

    void initNN();

    torch::Tensor lossFunction(torch::Tensor enhancement, torch::Tensor target);
    // double lostFunction(torch::Tensor potentialML, torch::Tensor target);

    void train();

    void dump();

    int nrxx = 1;
    int ninpt = 6;
    int nbatch = 10;

    std::vector<double> rho;
    // inputs
    std::vector<double> gamma;
    std::vector<double> gammanl;
    std::vector<double> p;
    std::vector<double> pnl;
    std::vector<double> q;
    std::vector<double> qnl;

    // target
    // std::vector<double> enhancement;
    torch::Tensor enhancement;
};

class OF_data : public torch::data::Dataset<OF_data>
{
private:
    torch::Tensor input;
    torch::Tensor target;
    
public:
    explicit OF_data(torch::Tensor &input, torch::Tensor &target)
    {
        this->input = input.clone();
        this->target = target.clone();
    }

    torch::data::Example<> get(size_t index) override 
    {
        return {this->input[index], this->target[index]};
    }

    torch::optional<size_t> size() const override 
    {
        return this->input.size(0);
    }
};
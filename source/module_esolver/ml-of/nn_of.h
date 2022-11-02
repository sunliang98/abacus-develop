#include <torch/torch.h>

struct NN_OF:torch::nn::Module{
    // three hidden layers and one output layer
    NN_OF();

    void setPara(int nrxx, int ninpt);

    void setData(std::vector<double> gamma, std::vector<double> gammanl, std::vector<double> p, std::vector<double> pnl, std::vector<double> q, std::vector<double> qnl);

    torch::Tensor forward(torch::Tensor inpt);

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};

    void getGradient();

    torch::Tensor getPotentail(torch::Tensor gradient, torch::Tensor inputs);

    torch::Tensor inputs;
    torch::Tensor F; // enhancement factor, output of NN
    torch::Tensor gradient;
    torch::Tensor potential;

    torch::Tensor normG;
    int nrxx;
    int ninpt;
};
// #ifndef NN_OF
// #define NN_OF

#include <torch/torch.h>

struct NN_OFImpl:torch::nn::Module{
    // three hidden layers and one output layer
    NN_OFImpl(
        int nrxx, 
        int ninpt, 
        int nnode,
        int nlayer
        );
    ~NN_OFImpl()
    {
        // delete[] this->fcs;
    }

    void setData(
        std::map<std::string, int> &input_index, 
        std::vector<double> &gamma, 
        std::vector<double> &p, 
        std::vector<double> &q, 
        std::vector<double> &gammanl, 
        std::vector<double> &pnl, 
        std::vector<double> &qnl,
        std::vector<double> &xi,
        std::vector<double> &tanhxi,
        std::vector<double> &tanhxi_l,
        std::vector<double> &tanhp,
        std::vector<double> &tanhq,
        std::vector<double> &tanh_pnl,
        std::vector<double> &tanh_qnl,
        std::vector<double> &tanhp_nl,
        std::vector<double> &tanhq_nl
    );

    void setData(
        std::map<std::string, int> &input_index, 
        torch::Tensor gamma, 
        torch::Tensor p, 
        torch::Tensor q, 
        torch::Tensor gammanl, 
        torch::Tensor pnl, 
        torch::Tensor qnl,
        torch::Tensor xi,
        torch::Tensor tanhxi,
        torch::Tensor tanhxi_nl,
        torch::Tensor tanhp,
        torch::Tensor tanhq,
        torch::Tensor tanh_pnl,
        torch::Tensor tanh_qnl,
        torch::Tensor tanhp_nl,
        torch::Tensor tanhq_nl
    );

    torch::Tensor forward(torch::Tensor inpt);

    // torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr};
    // torch::nn::Linear fcs[5] = {fc1, fc2, fc3, fc4, fc5};

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};

    torch::Tensor inputs;
    torch::Tensor F; // enhancement factor, output of NN
    torch::Tensor gradient;
    torch::Tensor potential;

    int nrxx = 10;
    int ninpt = 6;
    int nnode = 10;
    int nlayer = 3;
    int nfc = 4;
};
TORCH_MODULE(NN_OF);

// #endif
#include "nn_of.h"

NN_OFImpl::NN_OFImpl(int nrxx, int ninpt)
{
    this->nrxx = nrxx;
    this->ninpt = ninpt;

    this->inputs = torch::zeros({this->nrxx, this->ninpt});
    this->F = torch::zeros({this->nrxx, 1});
    this->gradient = torch::zeros({this->nrxx, this->ninpt});
    this->potential = torch::zeros({this->nrxx, 1});

    fc1 = register_module("fc1", torch::nn::Linear(ninpt, 10));
    fc2 = register_module("fc2", torch::nn::Linear(10, 10));
    fc3 = register_module("fc3", torch::nn::Linear(10, 10));
    fc4 = register_module("fc4", torch::nn::Linear(10, 1));
}

void NN_OFImpl::setData(
    std::map<std::string, int> &input_index, 
    std::vector<double> &gamma, 
    std::vector<double> &gammanl, 
    std::vector<double> &p, 
    std::vector<double> &pnl, 
    std::vector<double> &q, 
    std::vector<double> &qnl)
{
    if (input_index["gamma"] >= 0) this->inputs.index({"...", input_index["gamma"]}) = torch::tensor(gamma);
    if (input_index["p"] >= 0) this->inputs.index({"...", input_index["p"]}) = torch::tensor(p);
    if (input_index["q"] >= 0) this->inputs.index({"...", input_index["q"]}) = torch::tensor(q);
    if (input_index["gammanl"] >= 0) this->inputs.index({"...", input_index["gammanl"]}) = torch::tensor(gammanl);
    if (input_index["pnl"] >= 0) this->inputs.index({"...", input_index["pnl"]}) = torch::tensor(pnl);
    if (input_index["qnl"] >= 0) this->inputs.index({"...", input_index["qnl"]}) = torch::tensor(qnl);
}

void NN_OFImpl::setData(
    std::map<std::string, int> &input_index, 
    torch::Tensor &gamma, 
    torch::Tensor &gammanl, 
    torch::Tensor &p, 
    torch::Tensor &pnl, 
    torch::Tensor &q, 
    torch::Tensor &qnl
)
{
    if (input_index["gamma"] >= 0) this->inputs.index({"...", input_index["gamma"]}) = gamma.clone();
    if (input_index["p"] >= 0) this->inputs.index({"...", input_index["p"]}) = p.clone();
    if (input_index["q"] >= 0) this->inputs.index({"...", input_index["q"]}) = q.clone();
    if (input_index["gammanl"] >= 0) this->inputs.index({"...", input_index["gammanl"]}) = gammanl.clone();
    if (input_index["pnl"] >= 0) this->inputs.index({"...", input_index["pnl"]}) = pnl.clone();
    if (input_index["qnl"] >= 0) this->inputs.index({"...", input_index["qnl"]}) = qnl.clone();
}

torch::Tensor NN_OFImpl::forward(torch::Tensor inpt) // will inpt be changed? no
{
    inpt = torch::sigmoid(fc1->forward(inpt)); // covert data into (0,1)
    // dropout?
    inpt = torch::elu(fc2->forward(inpt));      // avoid overfitting (?)
    inpt = torch::elu(fc3->forward(inpt));      
    inpt = torch::elu(fc4->forward(inpt));  // ensure 0 < F_ML < 1 (?) no
    return inpt;
}

// void NN_OFImpl::getGradient()
// {
    // for (int ir = 0; ir < this->nrxx; ++ir)
    // {
    //     this->F[ir].backward();
    //     for (int j = 0; j < this->ninpt; ++j)
    //     {
    //         this->inputs[ir][j].grad();
    //     }
    // }
    // // this->gradient = this->inputs.grad();
    // backward or autograd.grad??
// }

// int main()
// {
//     NN_OFImpl nnof;
//     nnof.inputs += 1.;
//     std::cout << "inputs\n" << nnof.inputs << std::endl;
//     for (int ir = 0; ir < nnof.nrxx; ++ir)
//     {
//         nnof.F[ir] = nnof.forward(nnof.inputs[ir]);
//     }
//     std::cout << "inputs after forward\n" << nnof.inputs << std::endl;
//     std::cout << "output\n" << nnof.F << std::endl;
// }
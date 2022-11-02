#include "nn_of.h"

NN_OF::NN_OF()
{
    this->nrxx = 2;
    this->ninpt = 6;

    fc1 = register_module("fc1", torch::nn::Linear(this->ninpt, 10));
    fc2 = register_module("fc2", torch::nn::Linear(10, 10));
    fc3 = register_module("fc3", torch::nn::Linear(10, 10));
    fc4 = register_module("fc4", torch::nn::Linear(10, 1));
}

void NN_OF::setPara(int nrxx, int ninpt)
{
    this->nrxx = nrxx;
    this->ninpt = ninpt;

    this->inputs = torch::zeros({this->nrxx, this->ninpt});
    this->F = torch::zeros({this->nrxx, 1});
    this->gradient = torch::zeros({this->nrxx, this->ninpt});
    this->potential = torch::zeros({this->nrxx, 1});
}

void NN_OF::setData(std::vector<double> gamma, std::vector<double> gammanl, std::vector<double> p, std::vector<double> pnl, std::vector<double> q, std::vector<double> qnl)
{
    for (int ir = 0; ir < this->nrxx; ++ir)
    {
        this->inputs[ir][0] = gamma[ir];
        this->inputs[ir][1] = gammanl[ir];
        this->inputs[ir][2] = p[ir];
        this->inputs[ir][3] = pnl[ir];
        this->inputs[ir][4] = q[ir];
        this->inputs[ir][5] = qnl[ir];
    }
}

torch::Tensor NN_OF::forward(torch::Tensor inpt) // will inpt be changed? no
{
    inpt = torch::sigmoid(fc1->forward(inpt)); // covert data into (0,1)
    // dropout?
    inpt = torch::elu(fc2->forward(inpt));      // avoid overfitting (?)
    inpt = torch::elu(fc3->forward(inpt));      
    inpt = torch::elu(fc4->forward(inpt));  // ensure 0 < F_ML < 1 (?)

    // this->F[i].backward();
    // this->gradient[i] = inpt.grad();
    
    return inpt;
}

void NN_OF::getGradient()
{
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
}

// int main()
// {
//     NN_OF nnof;
//     nnof.inputs += 1.;
//     std::cout << "inputs\n" << nnof.inputs << std::endl;
//     for (int ir = 0; ir < nnof.nrxx; ++ir)
//     {
//         nnof.F[ir] = nnof.forward(nnof.inputs[ir]);
//     }
//     std::cout << "inputs after forward\n" << nnof.inputs << std::endl;
//     std::cout << "output\n" << nnof.F << std::endl;
// }
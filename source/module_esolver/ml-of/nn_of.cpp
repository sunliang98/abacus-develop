#include "nn_of.h"

NN_OFImpl::NN_OFImpl(int nrxx, int nrxx_vali, int ninpt, int nnode, int nlayer, torch::Device device)
{
    this->nrxx = nrxx;
    this->nrxx_vali = nrxx_vali;
    this->ninpt = ninpt;
    this->nnode = nnode;
    std::cout << "nnode = " << this->nnode << std::endl;
    this->nlayer = nlayer;
    std::cout << "nlayer = " << this->nlayer << std::endl;
    this->nfc = nlayer + 1;

    this->inputs = torch::zeros({this->nrxx, this->ninpt}).to(device);
    this->F = torch::zeros({this->nrxx, 1}).to(device);
    if (nrxx_vali > 0) this->input_vali = torch::zeros({nrxx_vali, this->ninpt}).to(device);
    // this->gradient = torch::zeros({this->nrxx, this->ninpt}).to(device);
    // this->potential = torch::zeros({this->nrxx, 1}).to(device);


    // int ni = nnode;
    // int no = nnode;
    // std::string name = "fc";
    // for (int i = 0; i < this->nfc; ++i)
    // {
    //     if (i == 0)             ni = this->ninpt;
    //     else                    ni = this->nnode;

    //     if (i == this->nfc - 1) no = 1;
    //     else                    no = this->nnode;

    //     name = "fc" + std::to_string(i+1);
    //     fcs[i] = register_module(name, torch::nn::Linear(ni, no));
    // }

    fc1 = register_module("fc1", torch::nn::Linear(ninpt, nnode));
    fc2 = register_module("fc2", torch::nn::Linear(nnode, nnode));
    fc3 = register_module("fc3", torch::nn::Linear(nnode, nnode));
    fc4 = register_module("fc4", torch::nn::Linear(nnode, 1));

    // fc1 = register_module("fc1", torch::nn::Linear(ninpt, 15));
    // fc2 = register_module("fc2", torch::nn::Linear(15, 15));
    // fc3 = register_module("fc3", torch::nn::Linear(15, 15));
    // fc4 = register_module("fc4", torch::nn::Linear(15, 1));

    this->to(device);
}


torch::Tensor NN_OFImpl::forward(torch::Tensor inpt) // will inpt be changed? no
{
    // sigmoid-elu-elu original
    // inpt = torch::sigmoid(fc1->forward(inpt)); // covert data into (0,1)
    // inpt = torch::elu(fc2->forward(inpt));
    // inpt = torch::elu(fc3->forward(inpt));
    // inpt = fc4->forward(inpt);

    // elu  2023-02
    // inpt = torch::elu(fc1->forward(inpt));
    // inpt = torch::elu(fc2->forward(inpt));
    // inpt = torch::elu(fc3->forward(inpt));  
    // inpt = fc4->forward(inpt);

    // tanh 2023-03-01
    // for (int i = 0; i < this->nfc - 1; ++i)
    // {
    //     inpt = torch::tanh(this->fcs[i]->forward(inpt));
    // }
    // inpt = this->fcs[this->nfc - 1]->forward(inpt);
    inpt = torch::tanh(fc1->forward(inpt)); // covert data into (-1,1)
    inpt = torch::tanh(fc2->forward(inpt));
    inpt = torch::tanh(fc3->forward(inpt));
    // inpt = torch::softplus(fc4->forward(inpt));
    inpt = fc4->forward(inpt); // for feg = 3

    // softplus 2023-03-01 (failed)
    // inpt = torch::softplus(fc1->forward(inpt));
    // inpt = torch::softplus(fc2->forward(inpt));
    // inpt = torch::softplus(fc3->forward(inpt));
    // inpt = torch::softplus(fc4->forward(inpt));

    // tanh-elu-elu-softplus 2023-03-07
    // inpt = torch::tanh(fc1->forward(inpt));
    // inpt = torch::elu(fc2->forward(inpt));
    // inpt = torch::elu(fc3->forward(inpt));
    // inpt = torch::softplus(fc4->forward(inpt));

    // dropout?
    // inpt = torch::relu(fc4->forward(inpt));
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
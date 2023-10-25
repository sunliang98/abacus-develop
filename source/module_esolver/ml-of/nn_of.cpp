#include "nn_of.h"

NN_OFImpl::NN_OFImpl(int nrxx, int ninpt, int nnode, int nlayer)
{
    this->nrxx = nrxx;
    this->ninpt = ninpt;
    this->nnode = nnode;
    std::cout << "nnode = " << this->nnode << std::endl;
    this->nlayer = nlayer;
    std::cout << "nlayer = " << this->nlayer << std::endl;
    this->nfc = nlayer + 1;

    this->inputs = torch::zeros({this->nrxx, this->ninpt});
    this->F = torch::zeros({this->nrxx, 1});
    this->gradient = torch::zeros({this->nrxx, this->ninpt});
    this->potential = torch::zeros({this->nrxx, 1});


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
}

void NN_OFImpl::setData(
    std::map<std::string, int> &input_index, 
    std::vector<double> &gamma, 
    std::vector<double> &p, 
    std::vector<double> &q, 
    std::vector<double> &gammanl, 
    std::vector<double> &pnl, 
    std::vector<double> &qnl,
    std::vector<double> &xi,
    std::vector<double> &tanhxi,
    std::vector<double> &tanhxi_nl,
    std::vector<double> &tanhp,
    std::vector<double> &tanhq,
    std::vector<double> &tanh_pnl,
    std::vector<double> &tanh_qnl,
    std::vector<double> &tanhp_nl,
    std::vector<double> &tanhq_nl
)
{
    if (input_index["gamma"] >= 0) this->inputs.index({"...", input_index["gamma"]}) = torch::tensor(gamma);
    if (input_index["p"] >= 0) this->inputs.index({"...", input_index["p"]}) = torch::tensor(p);
    if (input_index["q"] >= 0) this->inputs.index({"...", input_index["q"]}) = torch::tensor(q);
    if (input_index["gammanl"] >= 0) this->inputs.index({"...", input_index["gammanl"]}) = torch::tensor(gammanl);
    if (input_index["pnl"] >= 0) this->inputs.index({"...", input_index["pnl"]}) = torch::tensor(pnl);
    if (input_index["qnl"] >= 0) this->inputs.index({"...", input_index["qnl"]}) = torch::tensor(qnl);
    if (input_index["xi"] >= 0) this->inputs.index({"...", input_index["xi"]}) = torch::tensor(xi);
    if (input_index["tanhxi"] >= 0) this->inputs.index({"...", input_index["tanhxi"]}) = torch::tensor(tanhxi);
    if (input_index["tanhxi_nl"] >= 0) this->inputs.index({"...", input_index["tanhxi_nl"]}) = torch::tensor(tanhxi_nl);
    if (input_index["tanhp"] >= 0) this->inputs.index({"...", input_index["tanhp"]}) = torch::tensor(tanhp);
    if (input_index["tanhq"] >= 0) this->inputs.index({"...", input_index["tanhq"]}) = torch::tensor(tanhq);
    if (input_index["tanh_pnl"] >= 0) this->inputs.index({"...", input_index["tanh_pnl"]}) = torch::tensor(tanh_pnl);
    if (input_index["tanh_qnl"] >= 0) this->inputs.index({"...", input_index["tanh_qnl"]}) = torch::tensor(tanh_qnl);
    if (input_index["tanhp_nl"] >= 0) this->inputs.index({"...", input_index["tanhp_nl"]}) = torch::tensor(tanhp_nl);
    if (input_index["tanhq_nl"] >= 0) this->inputs.index({"...", input_index["tanhq_nl"]}) = torch::tensor(tanhq_nl);
}

void NN_OFImpl::setData(
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
)
{
    if (input_index["gamma"] >= 0) this->inputs.index({"...", input_index["gamma"]}) = gamma.clone();
    if (input_index["p"] >= 0) this->inputs.index({"...", input_index["p"]}) = p.clone();
    if (input_index["q"] >= 0) this->inputs.index({"...", input_index["q"]}) = q.clone();
    if (input_index["gammanl"] >= 0) this->inputs.index({"...", input_index["gammanl"]}) = gammanl.clone();
    if (input_index["pnl"] >= 0) this->inputs.index({"...", input_index["pnl"]}) = pnl.clone();
    if (input_index["qnl"] >= 0) this->inputs.index({"...", input_index["qnl"]}) = qnl.clone();
    if (input_index["xi"] >= 0) this->inputs.index({"...", input_index["xi"]}) = xi.clone();
    if (input_index["tanhxi"] >= 0) this->inputs.index({"...", input_index["tanhxi"]}) = tanhxi.clone();
    if (input_index["tanhxi_nl"] >= 0) this->inputs.index({"...", input_index["tanhxi_nl"]}) = tanhxi_nl.clone();
    if (input_index["tanhp"] >= 0) this->inputs.index({"...", input_index["tanhp"]}) = tanhp.clone();
    if (input_index["tanhq"] >= 0) this->inputs.index({"...", input_index["tanhq"]}) = tanhq.clone();
    if (input_index["tanh_pnl"] >= 0) this->inputs.index({"...", input_index["tanh_pnl"]}) = tanh_pnl.clone();
    if (input_index["tanh_qnl"] >= 0) this->inputs.index({"...", input_index["tanh_qnl"]}) = tanh_qnl.clone();
    if (input_index["tanhp_nl"] >= 0) this->inputs.index({"...", input_index["tanhp_nl"]}) = tanhp_nl.clone();
    if (input_index["tanhq_nl"] >= 0) this->inputs.index({"...", input_index["tanhq_nl"]}) = tanhq_nl.clone();
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
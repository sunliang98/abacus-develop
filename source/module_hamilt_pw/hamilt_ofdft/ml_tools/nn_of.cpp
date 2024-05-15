#include "nn_of.h"

NN_OFImpl::NN_OFImpl(int nrxx, int nrxx_vali, int ninpt, int nnode, int nlayer, int n_rho_out, int n_max, int m_right, int m_left, torch::Device device)
{
    this->nrxx = nrxx;
    this->nrxx_vali = nrxx_vali;
    this->ninpt = ninpt;
    this->nnode = nnode;
    std::cout << "nnode = " << this->nnode << std::endl;
    this->nlayer = nlayer;
    std::cout << "nlayer = " << this->nlayer << std::endl;
    this->nfc = nlayer + 1;
    this->n_rho_out = n_rho_out;
    std::cout << "n_rho_out = " << this->n_rho_out << std::endl;
    this->n_max = n_max;
    std::cout << "n_max = " << this->n_max << std::endl;
    this->m_right = m_right;
    std::cout << "m_right = " << this->m_right << std::endl;
    this->m_left = m_left;
    std::cout << "m_left = " << this->m_left << std::endl;

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


    rho_fc1 = register_module("rho_fc1", torch::nn::Linear(ninpt, 40));
    rho_fc2 = register_module("rho_fc2", torch::nn::Linear(40, 80));
    rho_fc3 = register_module("rho_fc3", torch::nn::Linear(80, 160));
    rho_fc4 = register_module("rho_fc4", torch::nn::Linear(160, n_rho_out));


    r_fc1 = register_module("r_fc1", torch::nn::Linear(1, 20));
    r_fc2 = register_module("r_fc2", torch::nn::Linear(20, 40));
    // r_fc3 = register_module("r_fc3", torch::nn::Linear(40, nnode));
    r_fc4 = register_module("r_fc4", torch::nn::Linear(40, this->m_right));

    
    tot_fc1 = register_module("tot_fc1", torch::nn::Linear(n_rho_out + this->m_right * this->m_left, 200));
    tot_fc2 = register_module("tot_fc2", torch::nn::Linear(200, 200));
    tot_fc3 = register_module("tot_fc3", torch::nn::Linear(200, 200));
    tot_fc4 = register_module("tot_fc4", torch::nn::Linear(200, 1));

    // rho_fc1 = register_module("rho_fc1", torch::nn::Linear(ninpt, 15));
    // rho_fc2 = register_module("rho_fc2", torch::nn::Linear(15, 15));
    // rho_fc3 = register_module("rho_fc3", torch::nn::Linear(15, 15));
    // rho_fc4 = register_module("rho_fc4", torch::nn::Linear(15, 1));

    this->to(device);
}


torch::Tensor NN_OFImpl::forward(torch::Tensor inpt, torch::Tensor R) // will inpt be changed? no
{
    // sigmoid-elu-elu original
    // inpt = torch::sigmoid(rho_fc1->forward(inpt)); // covert data into (0,1)
    // inpt = torch::elu(rho_fc2->forward(inpt));
    // inpt = torch::elu(rho_fc3->forward(inpt));
    // inpt = rho_fc4->forward(inpt);

    // elu  2023-02
    // inpt = torch::elu(rho_fc1->forward(inpt));
    // inpt = torch::elu(rho_fc2->forward(inpt));
    // inpt = torch::elu(rho_fc3->forward(inpt));  
    // inpt = rho_fc4->forward(inpt);

    // tanh 2023-03-01
    // for (int i = 0; i < this->nfc - 1; ++i)
    // {
    //     inpt = torch::tanh(this->fcs[i]->forward(inpt));
    // }
    // inpt = this->fcs[this->nfc - 1]->forward(inpt);
    // std::cout << "rho term" << std::endl;
    // rho term
    inpt = torch::tanh(rho_fc1->forward(inpt)); // covert data into (-1,1)
    inpt = torch::tanh(rho_fc2->forward(inpt));
    inpt = torch::tanh(rho_fc3->forward(inpt));
    inpt = torch::tanh(rho_fc4->forward(inpt));
    // std::cout << "inpt size = " << inpt.sizes() << std::endl;
    // inpt = torch::softplus(rho_fc4->forward(inpt));
    // inpt = rho_fc4->forward(inpt); // for feg = 3

    torch::Tensor G = R.index({"...", 0}).reshape({-1, 1}).contiguous();
    G = torch::tanh(r_fc1->forward(G));
    G = torch::tanh(r_fc2->forward(G));
    // G = torch::tanh(r_fc3->forward(G));
    G = torch::tanh(r_fc4->forward(G)).contiguous().view({-1, this->n_max, this->m_right}); // (nx * n_max) * m_right
    G = torch::bmm(R.transpose(1,2), G); // R^T * G
    G = torch::bmm((torch::slice(G, 2, 0, this->m_left)).transpose(1,2), G); // D = G<^T R R^T G

    // tot term
    inpt = torch::cat({inpt, G.contiguous().view({-1, this->m_right * this->m_left})}, 1);
    // std::cout << "inpt size = " << inpt.sizes() << std::endl;
    inpt = torch::tanh(tot_fc1->forward(inpt));
    inpt = torch::tanh(tot_fc2->forward(inpt));
    inpt = torch::tanh(tot_fc3->forward(inpt));
    inpt = tot_fc4->forward(inpt); // for feg = 3

    // softplus 2023-03-01 (failed)
    // inpt = torch::softplus(rho_fc1->forward(inpt));
    // inpt = torch::softplus(rho_fc2->forward(inpt));
    // inpt = torch::softplus(rho_fc3->forward(inpt));
    // inpt = torch::softplus(rho_fc4->forward(inpt));

    // tanh-elu-elu-softplus 2023-03-07
    // inpt = torch::tanh(rho_fc1->forward(inpt));
    // inpt = torch::elu(rho_fc2->forward(inpt));
    // inpt = torch::elu(rho_fc3->forward(inpt));
    // inpt = torch::softplus(rho_fc4->forward(inpt));

    // dropout?
    // inpt = torch::relu(rho_fc4->forward(inpt));
    return inpt;
}

torch::Tensor NN_OFImpl::get_D(torch::Tensor &R)
{
    std::cout << "R.requires_grad() = " << R.requires_grad() << std::endl;
    torch::Tensor G = R.index({"...", 0}).reshape({-1, 1}).contiguous();

    G = torch::tanh(r_fc1->forward(G));
    G = torch::tanh(r_fc2->forward(G));
    // G = torch::tanh(r_fc3->forward(G));
    G = torch::tanh(r_fc4->forward(G)).contiguous().view({this->nrxx, this->n_max, this->m_right}); // (nx * n_max) * m_right
    G = torch::bmm(R.transpose(1,2), G); // R^T * G
    G = torch::bmm((torch::slice(G, 2, 0, this->m_left)).transpose(1,2), G); // D = G<^T R R^T G
    return G.contiguous().view({this->nrxx, this->m_right * this->m_left});
}

torch::Tensor NN_OFImpl::forward_with_D(torch::Tensor inpt, torch::Tensor &D)
{
    inpt = torch::tanh(rho_fc1->forward(inpt)); // covert data into (-1,1)
    inpt = torch::tanh(rho_fc2->forward(inpt));
    inpt = torch::tanh(rho_fc3->forward(inpt));
    inpt = torch::tanh(rho_fc4->forward(inpt));

    inpt = torch::cat({inpt, D}, 1);
    // std::cout << "inpt size = " << inpt.sizes() << std::endl;
    inpt = torch::tanh(tot_fc1->forward(inpt));
    inpt = torch::tanh(tot_fc2->forward(inpt));
    inpt = torch::tanh(tot_fc3->forward(inpt));
    inpt = tot_fc4->forward(inpt); // for feg = 3

    // softplus 2023-03-01 (failed)
    // inpt = torch::softplus(rho_fc1->forward(inpt));
    // inpt = torch::softplus(rho_fc2->forward(inpt));
    // inpt = torch::softplus(rho_fc3->forward(inpt));
    // inpt = torch::softplus(rho_fc4->forward(inpt));

    // tanh-elu-elu-softplus 2023-03-07
    // inpt = torch::tanh(rho_fc1->forward(inpt));
    // inpt = torch::elu(rho_fc2->forward(inpt));
    // inpt = torch::elu(rho_fc3->forward(inpt));
    // inpt = torch::softplus(rho_fc4->forward(inpt));

    // dropout?
    // inpt = torch::relu(rho_fc4->forward(inpt));
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
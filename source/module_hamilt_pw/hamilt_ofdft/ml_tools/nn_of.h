#ifndef NN_OF_H
#define NN_OF_H

#include <torch/torch.h>

struct NN_OFImpl:torch::nn::Module{
    // three hidden layers and one output layer
    NN_OFImpl(
        int nrxx, 
        int nrxx_vali, 
        int ninpt, 
        int nnode,
        int nlayer,
        int n_rho_out,
        int n_max,
        int m_right,
        int m_left,
        torch::Device device
        );
    ~NN_OFImpl()
    {
        // delete[] this->fcs;
    };


    template <class T>
    void set_data(
        T *data,
        const std::vector<std::string> &descriptor_type,
        const std::vector<int> &kernel_index,
        torch::Tensor &nn_input
    )
    {
        if (data->nx_tot <= 0) return;
        for (int i = 0; i < descriptor_type.size(); ++i)
        {
            nn_input.index({"...", i}) = data->get_data(descriptor_type[i], kernel_index[i]);
        }
    }

    torch::Tensor forward(torch::Tensor inpt, torch::Tensor R);
    torch::Tensor get_D(torch::Tensor &R);
    torch::Tensor forward_with_D(torch::Tensor inpt, torch::Tensor &D);

    // torch::nn::Linear rho_fc1{nullptr}, rho_fc2{nullptr}, rho_fc3{nullptr}, rho_fc4{nullptr}, fc5{nullptr};
    // torch::nn::Linear fcs[5] = {rho_fc1, rho_fc2, rho_fc3, rho_fc4, fc5};

    torch::nn::Linear rho_fc1{nullptr}, rho_fc2{nullptr}, rho_fc3{nullptr}, rho_fc4{nullptr};
    torch::nn::Linear r_fc1{nullptr}, r_fc2{nullptr}, r_fc3{nullptr}, r_fc4{nullptr};
    torch::nn::Linear tot_fc1{nullptr}, tot_fc2{nullptr}, tot_fc3{nullptr}, tot_fc4{nullptr};

    torch::Tensor inputs;
    torch::Tensor R; // r_matrix
    torch::Tensor input_vali;
    torch::Tensor R_vali;
    torch::Tensor F; // enhancement factor, output of NN
    // torch::Tensor gradient;
    // torch::Tensor potential;



    int nrxx = 10;
    int nrxx_vali = 0;
    int ninpt = 6;
    int nnode = 10;
    int nlayer = 3;
    int nfc = 4;
    int n_rho_out = 20;
    // atomic position D=G<^T R R^T G
    // G<(n_max * m_left) R(n_max * 4) G(n_max * m_right)
    int n_max = 0;
    int m_right = 0;
    int m_left = 0; // m_left <= m_right
};
TORCH_MODULE(NN_OF);

#endif
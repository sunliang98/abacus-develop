#ifndef DATA_H
#define DATA_H

#include "./input.h"

#include <torch/torch.h>

class Data
{
    // --------- load the data from .npy files ------
  public:
    int nx = 0;
    int nx_tot = 0;

    torch::Tensor rho;
    // inputs
    torch::Tensor gamma;
    torch::Tensor p;
    torch::Tensor q;
    torch::Tensor gammanl;
    torch::Tensor pnl;
    torch::Tensor qnl;
    torch::Tensor nablaRho;
    // new parameters 2023-02-14
    torch::Tensor xi;
    torch::Tensor tanhxi;
    torch::Tensor tanhxi_nl; // 2023-03-20
    torch::Tensor tanhp;
    torch::Tensor tanhq;
    torch::Tensor tanh_pnl;
    torch::Tensor tanh_qnl;
    torch::Tensor tanhp_nl;
    torch::Tensor tanhq_nl;
    // target
    torch::Tensor enhancement;
    torch::Tensor pauli;
    torch::Tensor enhancement_mean;
    torch::Tensor tau_mean; // mean Pauli energy
    torch::Tensor pauli_mean;

    void loadData(Input &input, const int ndata, std::string *dir, const torch::Device device);

  private:
    void initData(const int ndata, const int fftdim, const torch::Device device);
    void loadData_(Input &input, const int ndata, const int fftdim, std::string *dir);

  public:
    void loadTensor(std::string file,
                    std::vector<long unsigned int> cshape,
                    bool fortran_order,
                    std::vector<double> &container,
                    const int index,
                    const int fftdim,
                    torch::Tensor &data);
    // -------- dump Tensor into .npy files ---------
    void dumpTensor(const torch::Tensor &data, std::string filename, int nx);
};
#endif
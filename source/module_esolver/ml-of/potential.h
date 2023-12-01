#ifndef POTENTIAL_H
#define POTENTIAL_H

#include <torch/torch.h>
#include "./input.h"

class Potential{

public:
    void init(const Input &input, const int fftdim_in, const std::map<std::string, int> &nn_input_index_in)
    {
        this->fftdim = fftdim_in;
        this->nn_input_index = nn_input_index_in;
        this->ml_gamma = input.ml_gamma;
        this->ml_p = input.ml_p;
        this->ml_q = input.ml_q;
        this->ml_gammanl = input.ml_gammanl;
        this->ml_pnl = input.ml_pnl;
        this->ml_qnl = input.ml_qnl;
        this->ml_xi = input.ml_xi;
        this->ml_tanhxi = input.ml_tanhxi;
        this->ml_tanhxi_nl = input.ml_tanhxi_nl;
        this->ml_tanhp = input.ml_tanhp;
        this->ml_tanhq = input.ml_tanhq;
        this->ml_tanh_pnl = input.ml_tanh_pnl;
        this->ml_tanh_qnl = input.ml_tanh_qnl;
        this->ml_tanhp_nl = input.ml_tanhp_nl;
        this->ml_tanhq_nl = input.ml_tanhq_nl;
        this->chi_xi = input.chi_xi;
        this->chi_p = input.chi_p;
        this->chi_q = input.chi_q;
        this->chi_pnl = input.chi_pnl;
        this->chi_qnl = input.chi_qnl;
    }

    int fftdim = 0;
    std::map<std::string, int> nn_input_index;

    bool ml_gamma = false;
    bool ml_p = false;
    bool ml_q = false;
    bool ml_gammanl = false;
    bool ml_pnl = false;
    bool ml_qnl = false;
    bool ml_xi = false;
    bool ml_tanhxi = false;
    bool ml_tanhxi_nl = false; // 2023-03-20
    bool ml_tanhp = false;
    bool ml_tanhq = false;
    bool ml_tanh_pnl = false;
    bool ml_tanh_qnl = false;
    bool ml_tanhp_nl = false;
    bool ml_tanhq_nl = false;
    double chi_xi = 1.;
    double chi_p = 1.;
    double chi_q = 1.;
    double chi_pnl = 1.;
    double chi_qnl = 1.;

    torch::Tensor getPot(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &tauTF,
        const torch::Tensor &gamma,
        const torch::Tensor &p,
        const torch::Tensor &q,
        const torch::Tensor &xi,
        const torch::Tensor &tanhxi,
        const torch::Tensor &tanhxi_nl,
        const torch::Tensor &tanhp,
        const torch::Tensor &tanhq,
        const torch::Tensor &tanh_pnl,
        const torch::Tensor &tanh_qnl,
        const torch::Tensor &F,
        const torch::Tensor &gradient,
        const torch::Tensor &kernel,
        const std::vector<torch::Tensor> &grid,
        const torch::Tensor &gg
    );
private:

    torch::Tensor potGammaTerm(
        const torch::Tensor &gamma,
        const torch::Tensor &gradient
    );
    torch::Tensor potPTerm1(
        const torch::Tensor &p,
        const torch::Tensor &gradient
    );
    torch::Tensor potQTerm1(
        const torch::Tensor &q,
        const torch::Tensor &gradient
    );
    torch::Tensor potGammanlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &gamma,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient
    );
    torch::Tensor potPPnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &p,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const std::vector<torch::Tensor> &grid
    );
    torch::Tensor potQQnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &q,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const torch::Tensor &gg    
    );


    torch::Tensor potXiTerm1(
        const torch::Tensor &xi,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhxiTerm1(
        const torch::Tensor &xi,
        const torch::Tensor &tanhxi,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhpTerm1(
        const torch::Tensor &p,
        const torch::Tensor &tanhp,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhqTerm1(
        const torch::Tensor &q,
        const torch::Tensor &tanhq,
        const torch::Tensor &gradient
    );
    torch::Tensor potXinlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhxinlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &tanhxi,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhxi_nlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &xi,
        const torch::Tensor &tanhxi,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhpTanh_pnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &p,
        const torch::Tensor &tanhp,
        const torch::Tensor &tanh_pnl,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const std::vector<torch::Tensor> &grid
    );
    torch::Tensor potTanhqTanh_qnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &q,
        const torch::Tensor &tanhq,
        const torch::Tensor &tanh_qnl,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const torch::Tensor &gg
    );
    torch::Tensor potTanhpTanhp_nlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &p,
        const torch::Tensor &tanhp,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const std::vector<torch::Tensor> &grid
    );
    torch::Tensor potTanhqTanhq_nlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &q,
        const torch::Tensor &tanhq,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const torch::Tensor &gg
    );

    // Tools for getting potential
    torch::Tensor divergence(
        const torch::Tensor &input,
        const std::vector<torch::Tensor> &grid
    );
    torch::Tensor Laplacian(
        const torch::Tensor &input,
        const torch::Tensor &gg
    );
    torch::Tensor dtanh(
        const torch::Tensor &tanhx,
        const double chi
    );

    const double cTF = 3.0/10.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0) * 2; // 10/3*(3*pi^2)^{2/3}, multiply by 2 to convert unit from Hartree to Ry, finally in Ry*Bohr^(-2)
    const double pqcoef = 1.0 / (4.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0)); // coefficient of p and q
};
#endif
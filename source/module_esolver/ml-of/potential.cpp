#include "./potential.h"

torch::Tensor Potential::getPot(
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
)
{
    return tauTF / rho * (5./3. * F + this->potGammaTerm(gamma, gradient) + this->potPTerm1(p, gradient) + this->potQTerm1(q, gradient)
            + this->potXiTerm1(xi, gradient) + this->potTanhxiTerm1(xi, tanhxi, gradient)
            + this->potTanhpTerm1(p, tanhp, gradient) + this->potTanhqTerm1(q, tanhq, gradient))
            + this->potGammanlTerm(rho, gamma, kernel, tauTF, gradient)
            + this->potPPnlTerm(rho, nablaRho, p, kernel, tauTF, gradient, grid)
            + this->potQQnlTerm(rho, q, kernel, tauTF, gradient, gg)
            + this->potXinlTerm(rho, kernel, tauTF, gradient) + this->potTanhxinlTerm(rho, tanhxi, kernel, tauTF, gradient)
            + this->potTanhxi_nlTerm(rho, xi, tanhxi, kernel, tauTF, gradient)
            + this->potTanhpTanh_pnlTerm(rho, nablaRho, p, tanhp, tanh_pnl, kernel, tauTF, gradient, grid)
            + this->potTanhqTanh_qnlTerm(rho, q, tanhq, tanh_qnl, kernel, tauTF, gradient, gg)
            + this->potTanhpTanhp_nlTerm(rho, nablaRho, p, tanhp, kernel, tauTF, gradient, grid)
            + this->potTanhqTanhq_nlTerm(rho, q, tanhq, kernel, tauTF, gradient, gg);
}

torch::Tensor Potential::potGammaTerm(
    const torch::Tensor &gamma,
    const torch::Tensor &gradient
)
{
    // std::cout << "potGammaTerm" << std::endl;
    return (this->ml_gamma) ? 1./3. * gamma * gradient.index({"...", this->nn_input_index["gamma"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(gamma);
}

torch::Tensor Potential::potPTerm1(
    const torch::Tensor &p,
    const torch::Tensor &gradient
)
{
    // std::cout << "potPTerm1" << std::endl;
    return (this->ml_p) ? - 8./3. * p * gradient.index({"...", this->nn_input_index["p"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(p);
}

torch::Tensor Potential::potQTerm1(
    const torch::Tensor &q,
    const torch::Tensor &gradient
)
{
    // std::cout << "potQTerm1" << std::endl;
    return (this->ml_q) ? - 5./3. * q * gradient.index({"...", this->nn_input_index["q"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(q);
}

torch::Tensor Potential::potGammanlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &gamma,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient
)
{
    // std::cout << "potGmmamnlTerm" << std::endl;
    if (!this->ml_gammanl) return torch::zeros_like(gamma);
    else return 1./3. * gamma / rho * torch::real(torch::fft::ifftn(torch::fft::fftn(gradient.index({"...", this->nn_input_index["gammanl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) * tauTF) * kernel));
}

torch::Tensor Potential::potPPnlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &nablaRho,
    const torch::Tensor &p,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient,
    const std::vector<torch::Tensor> &grid
)
{
    // std::cout << "potPPnlTerm" << std::endl;
    if (!this->ml_p && !this->ml_pnl) return torch::zeros_like(p);
    torch::Tensor dFdpnl_nl = torch::zeros_like(p);
    if (this->ml_pnl) dFdpnl_nl = torch::real(torch::fft::ifftn(torch::fft::fftn(gradient.index({"...", this->nn_input_index["pnl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) * tauTF) * kernel));

    torch::Tensor temp = torch::zeros_like(nablaRho);
    for (int i = 0; i < 3; ++i)
    {
        temp[i] = (this->ml_p) ? -3./20. * gradient.index({"...", this->nn_input_index["p"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) * nablaRho[i] / rho * /*Ha to Ry*/2. : torch::zeros_like(nablaRho[i]);
        if (this->ml_pnl) temp[i] += - this->pqcoef * 2. * nablaRho[i] / torch::pow(rho, 8./3.) * dFdpnl_nl;
    }
    // std::cout << torch::slice(temp[0][0][0], 0, 0, 10);
    torch::Tensor result = this->divergence(temp, grid);

    if (this->ml_pnl) result += -8./3. * p / rho * dFdpnl_nl;
    // std::cout << torch::slice(result[0][0], 0, 20) << std::endl;
    
    // std::cout << "potPPnlTerm done" << std::endl;
    return result;
}

torch::Tensor Potential::potQQnlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &q,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient,
    const torch::Tensor &gg    
)
{
    // std::cout << "potQQnlTerm" << std::endl;
    if (!this->ml_q && !this->ml_qnl) return torch::zeros_like(q);
    torch::Tensor dFdqnl_nl = torch::zeros_like(q);
    if (this->ml_qnl) dFdqnl_nl = torch::real(torch::fft::ifftn(torch::fft::fftn(gradient.index({"...", this->nn_input_index["qnl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) * tauTF) * kernel));

    torch::Tensor temp = (this->ml_q) ? 3./40. * gradient.index({"...", this->nn_input_index["q"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) * /*Ha2Ry*/2. : torch::zeros_like(q);
    if (this->ml_qnl) temp += this->pqcoef / torch::pow(rho, 5./3.) * dFdqnl_nl;
    torch::Tensor result = this->Laplacian(temp, gg);

    if (this->ml_qnl) result += - 5./3. * q / rho * dFdqnl_nl;

    // std::cout << "potQQnlTerm done" << std::endl;
    return result;
}

torch::Tensor Potential::potXiTerm1(
    const torch::Tensor &xi,
    const torch::Tensor &gradient
)
{
    return (this->ml_xi) ? -1./3. * xi * gradient.index({"...", this->nn_input_index["xi"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(xi);
}

torch::Tensor Potential::potTanhxiTerm1(
    const torch::Tensor &xi,
    const torch::Tensor &tanhxi,
    const torch::Tensor &gradient
)
{
    return (this->ml_tanhxi) ? -1./3. * xi * this->dtanh(tanhxi, this->chi_xi)
                                * gradient.index({"...", this->nn_input_index["tanhxi"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(tanhxi);
}

torch::Tensor Potential::potTanhpTerm1(
    const torch::Tensor &p,
    const torch::Tensor &tanhp,
    const torch::Tensor &gradient
)
{
    return (this->ml_tanhp) ? - 8./3. * p * this->dtanh(tanhp, this->chi_p)
                                * gradient.index({"...", this->nn_input_index["tanhp"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(tanhp);
}

torch::Tensor Potential::potTanhqTerm1(
    const torch::Tensor &q,
    const torch::Tensor &tanhq,
    const torch::Tensor &gradient
)
{
    return (this->ml_tanhq) ? - 5./3. * q * this->dtanh(tanhq, this->chi_q)
                                * gradient.index({"...", this->nn_input_index["tanhq"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(tanhq);
}

torch::Tensor Potential::potXinlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient
)
{
    if (!this->ml_xi) return torch::zeros_like(rho);
    else return 1./3. * torch::pow(rho, -2./3.) * torch::real(torch::fft::ifftn(
                torch::fft::fftn(gradient.index({"...", this->nn_input_index["xi"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) 
                * tauTF * torch::pow(rho, -1./3.)) * kernel));
}

torch::Tensor Potential::potTanhxinlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &tanhxi,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient
)
{
    if (!this->ml_tanhxi) return torch::zeros_like(rho);
    else return 1./3. * torch::pow(rho, -2./3.) * torch::real(torch::fft::ifftn(
                torch::fft::fftn(gradient.index({"...", this->nn_input_index["tanhxi"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) 
                * this->dtanh(tanhxi, this->chi_xi)
                * tauTF * torch::pow(rho, -1./3.)) * kernel));
}

torch::Tensor Potential::potTanhxi_nlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &xi,
    const torch::Tensor &tanhxi,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient
)
{
    if (!this->ml_tanhxi_nl) return torch::zeros_like(rho);
    torch::Tensor dFdxi = torch::real(torch::fft::ifftn(torch::fft::fftn(
                          tauTF * gradient.index({"...", this->nn_input_index["tanhxi_nl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}))
                          * kernel)) 
                          * this->dtanh(tanhxi, this->chi_xi) * torch::pow(rho, -1./3.);
    return 1./3. * torch::pow(rho, -2./3.) 
           * (- xi * dFdxi
           + torch::real(torch::fft::ifftn(torch::fft::fftn(dFdxi) * kernel)));
}

torch::Tensor Potential::potTanhpTanh_pnlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &nablaRho,
    const torch::Tensor &p,
    const torch::Tensor &tanhp,
    const torch::Tensor &tanh_pnl,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient,
    const std::vector<torch::Tensor> &grid
)
{
    if (!this->ml_tanhp && !this->ml_tanh_pnl) return torch::zeros_like(tanhp);
    if (this->ml_tanhp_nl) return torch::zeros_like(tanhp);
    torch::Tensor dFdpnl_nl = torch::zeros_like(tanhp);
    if (this->ml_tanh_pnl){
        dFdpnl_nl = torch::real(torch::fft::ifftn(torch::fft::fftn(gradient.index({"...", this->nn_input_index["tanh_pnl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) 
                    * this->dtanh(tanh_pnl, this->chi_pnl)
                    * tauTF) * kernel));
    }

    torch::Tensor temp = torch::zeros_like(nablaRho);
    for (int i = 0; i < 3; ++i)
    {
        temp[i] = (this->ml_tanhp) ? -3./20. * gradient.index({"...", this->nn_input_index["tanhp"]}).reshape({this->fftdim, this->fftdim, this->fftdim})
                                     * this->dtanh(tanhp, this->chi_p) * nablaRho[i] / rho * /*Ha to Ry*/2. : torch::zeros_like(nablaRho[i]);
        if (this->ml_tanh_pnl) temp[i] += - this->pqcoef * 2. * nablaRho[i] / torch::pow(rho, 8./3.) * dFdpnl_nl;
    }
    torch::Tensor result = this->divergence(temp, grid);

    if (this->ml_tanh_pnl) result += -8./3. * p / rho * dFdpnl_nl;
    
    return result;
}

torch::Tensor Potential::potTanhqTanh_qnlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &q,
    const torch::Tensor &tanhq,
    const torch::Tensor &tanh_qnl,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient,
    const torch::Tensor &gg
)
{
    if (!this->ml_tanhq && !this->ml_tanh_qnl) return torch::zeros_like(tanhq);
    if (this->ml_tanhq_nl) return torch::zeros_like(tanhq);
    torch::Tensor dFdqnl_nl = torch::zeros_like(tanhq);
    if (this->ml_tanh_qnl){
        dFdqnl_nl = torch::real(torch::fft::ifftn(torch::fft::fftn(gradient.index({"...", this->nn_input_index["tanh_qnl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) 
                    * this->dtanh(tanh_qnl, this->chi_qnl)
                    * tauTF) * kernel));
    }

    torch::Tensor temp = (this->ml_tanhq) ? 3./40. * gradient.index({"...", this->nn_input_index["tanhq"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) 
                         * this->dtanh(tanhq, this->chi_q) * /*Ha2Ry*/2. : torch::zeros_like(q);
    if (this->ml_tanh_qnl) temp += this->pqcoef / torch::pow(rho, 5./3.) * dFdqnl_nl;
    torch::Tensor result = this->Laplacian(temp, gg);

    if (this->ml_tanh_qnl) result += - 5./3. * q / rho * dFdqnl_nl;

    return result;
}

torch::Tensor Potential::potTanhpTanhp_nlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &nablaRho,
    const torch::Tensor &p,
    const torch::Tensor &tanhp,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient,
    const std::vector<torch::Tensor> &grid
)
{
    if (!this->ml_tanhp_nl) return torch::zeros_like(tanhp);
    torch::Tensor dFdpnl_nl = torch::zeros_like(tanhp);
    if (this->ml_tanhp_nl){
        dFdpnl_nl = torch::real(torch::fft::ifftn(torch::fft::fftn(gradient.index({"...", this->nn_input_index["tanhp_nl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) 
                    * tauTF) * kernel))
                    * this->dtanh(tanhp, this->chi_p);
    }

    torch::Tensor temp = torch::zeros_like(nablaRho);
    for (int i = 0; i < 3; ++i)
    {
        temp[i] = (this->ml_tanhp) ? -3./20. * gradient.index({"...", this->nn_input_index["tanhp"]}).reshape({this->fftdim, this->fftdim, this->fftdim})
                                     * this->dtanh(tanhp, this->chi_p) * nablaRho[i] / rho * /*Ha to Ry*/2. : torch::zeros_like(nablaRho[i]);
        if (this->ml_tanhp_nl) temp[i] += - this->pqcoef * 2. * nablaRho[i] / torch::pow(rho, 8./3.) * dFdpnl_nl;
    }
    torch::Tensor result = this->divergence(temp, grid);

    if (this->ml_tanhp_nl) result += -8./3. * p / rho * dFdpnl_nl;
    
    return result;
}

torch::Tensor Potential::potTanhqTanhq_nlTerm(
    const torch::Tensor &rho,
    const torch::Tensor &q,
    const torch::Tensor &tanhq,
    const torch::Tensor &kernel,
    const torch::Tensor &tauTF,
    const torch::Tensor &gradient,
    const torch::Tensor &gg
)
{
    if (!this->ml_tanhq_nl) return torch::zeros_like(tanhq);
    torch::Tensor dFdqnl_nl = torch::zeros_like(tanhq);
    if (this->ml_tanhq_nl){
        dFdqnl_nl = torch::real(torch::fft::ifftn(torch::fft::fftn(gradient.index({"...", this->nn_input_index["tanhq_nl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) 
                    * tauTF) * kernel))
                    * this->dtanh(tanhq, this->chi_q);
    }

    torch::Tensor temp = (this->ml_tanhq) ? 3./40. * gradient.index({"...", this->nn_input_index["tanhq"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) 
                         * this->dtanh(tanhq, this->chi_q) * /*Ha2Ry*/2. : torch::zeros_like(q);
    if (this->ml_tanhq_nl) temp += this->pqcoef / torch::pow(rho, 5./3.) * dFdqnl_nl;
    torch::Tensor result = this->Laplacian(temp, gg);

    if (this->ml_tanhq_nl) result += - 5./3. * q / rho * dFdqnl_nl;

    return result;
}

torch::Tensor Potential::divergence(
    const torch::Tensor &input,
    const std::vector<torch::Tensor> &grid
)
{
    torch::Tensor result = torch::zeros_like(input[0]);
    // torch::Tensor img = torch::tensor({1.0j});
    // for (int i = 0; i < 3; ++i)
    // {
    //     result += torch::real(torch::fft::ifftn(torch::fft::fftn(input[i]) * grid[i] * img));
    // }
    for (int i = 0; i < 3; ++i)
    {
        result -= torch::imag(torch::fft::ifftn(torch::fft::fftn(input[i]) * grid[i]));
    }
    return result;
}

torch::Tensor Potential::Laplacian(
    const torch::Tensor &input,
    const torch::Tensor &gg
)
{
    return torch::real(torch::fft::ifftn(torch::fft::fftn(input) * - gg));
}

torch::Tensor Potential::dtanh(
    const torch::Tensor &tanhx,
    const double chi
)
{
    return (torch::ones_like(tanhx) - tanhx * tanhx) * chi;
    // return (1. - tanhx * tanhx) * chi;
}
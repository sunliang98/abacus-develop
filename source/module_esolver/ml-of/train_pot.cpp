#include "./train.h"

torch::Tensor Train::getPot(
    const torch::Tensor &rho,
    const torch::Tensor &nablaRho,
    const torch::Tensor &tauTF,
    const torch::Tensor &gamma,
    const torch::Tensor &p,
    const torch::Tensor &q,
    const torch::Tensor &F,
    const torch::Tensor &gradient,
    const torch::Tensor &kernel,
    const std::vector<torch::Tensor> &grid,
    const torch::Tensor &gg
)
{
    return tauTF / rho * (5./3. * F + this->potGammaTerm(gamma, gradient) + this->potPTerm1(p, gradient) + this->potQTerm1(q, gradient))
            + this->potGammanlTerm(rho, gamma, kernel, tauTF, gradient)
            + this->potPPnlTerm(rho, nablaRho, p, kernel, tauTF, gradient, grid)
            + this->potQQnlTerm(rho, q, kernel, tauTF, gradient, gg);
}

torch::Tensor Train::potGammaTerm(
    const torch::Tensor &gamma,
    const torch::Tensor &gradient
)
{
    // std::cout << "potGammaTerm" << std::endl;
    return (this->ml_gamma) ? 1./3. * gamma * gradient.index({"...", this->nn_input_index["gamma"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(gamma);
}

torch::Tensor Train::potPTerm1(
    const torch::Tensor &p,
    const torch::Tensor &gradient
)
{
    // std::cout << "potPTerm1" << std::endl;
    return (this->ml_p) ? - 8./3. * p * gradient.index({"...", this->nn_input_index["p"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(p);
}

torch::Tensor Train::potQTerm1(
    const torch::Tensor &q,
    const torch::Tensor &gradient
)
{
    // std::cout << "potQTerm1" << std::endl;
    return (this->ml_q) ? - 5./3. * q * gradient.index({"...", this->nn_input_index["q"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) : torch::zeros_like(q);
}

torch::Tensor Train::potGammanlTerm(
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

torch::Tensor Train::potPPnlTerm(
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
    if (this->ml_gammanl) dFdpnl_nl = torch::real(torch::fft::ifftn(torch::fft::fftn(gradient.index({"...", this->nn_input_index["pnl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) * tauTF) * kernel));

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

torch::Tensor Train::potQQnlTerm(
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
    if (this->ml_gammanl) dFdqnl_nl = torch::real(torch::fft::ifftn(torch::fft::fftn(gradient.index({"...", this->nn_input_index["qnl"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) * tauTF) * kernel));

    torch::Tensor temp = (this->ml_q) ? 3./40. * gradient.index({"...", this->nn_input_index["q"]}).reshape({this->fftdim, this->fftdim, this->fftdim}) * /*Ha2Ry*/2. : torch::zeros_like(q);
    if (this->ml_qnl) temp += this->pqcoef / torch::pow(rho, 5./3.) * dFdqnl_nl;
    torch::Tensor result = this->Laplacian(temp, gg);

    if (this->ml_qnl) result += - 5./3. * q / rho * dFdqnl_nl;

    // std::cout << "potQQnlTerm done" << std::endl;
    return result;
}

torch::Tensor Train::divergence(
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

torch::Tensor Train::Laplacian(
    const torch::Tensor &input,
    const torch::Tensor &gg
)
{
    return torch::real(torch::fft::ifftn(torch::fft::fftn(input) * - gg));
}
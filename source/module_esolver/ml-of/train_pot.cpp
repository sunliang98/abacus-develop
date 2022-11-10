#include "./train.h"

torch::Tensor Train::potGammaTerm(
    const torch::Tensor &gamma,
    const torch::Tensor &dFdgamma
)
{
    return (this->ml_gamma) ? 1./3. * gamma * dFdgamma : torch::zeros_like(gamma);
}

torch::Tensor Train::potPTerm1(
    const torch::Tensor &p,
    const torch::Tensor &dFdp
)
{
    return (ml_p) ? - 8./3. * p * dFdp : torch::zeros_like(p);
}

torch::Tensor Train::potQTerm1(
    const torch::Tensor &q,
    const torch::Tensor &dFdq
)
{
    return (ml_q) ? - 5./3. * q * dFdq : torch::zeros_like(q);
}
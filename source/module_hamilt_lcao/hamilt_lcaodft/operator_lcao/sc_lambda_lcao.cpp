#include "sc_lambda_lcao.h"
#include "module_hamilt_lcao/module_deltaspin/spin_constrain.h"
#include "module_parameter/parameter.h"
#include <algorithm>

namespace hamilt
{

// contribute to HR is not needed.
template <typename TK, typename TR>
void OperatorScLambda<OperatorLCAO<TK, TR>>::contributeHR()
{
    return;
}

// contribute to Hk
template <>
void OperatorScLambda<OperatorLCAO<std::complex<double>, std::complex<double>>>::contributeHk(int ik)
{
    ModuleBase::TITLE("OperatorScLambda", "contributeHk");
    ModuleBase::timer::tick("OperatorScLambda", "contributeHk");
    SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>& sc
        = SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::getScInstance();
    std::vector<std::complex<double>> h_lambda(this->hsk->get_pv()->nloc);
    std::fill(h_lambda.begin(), h_lambda.end(), std::complex<double>(0, 0));
    sc.cal_h_lambda(&h_lambda[0], this->hsk->get_sk(), ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER(PARAM.inp.ks_solver), this->isk[ik]);
    std::complex<double>* hk = this->hsk->get_hk();
    for (int irc = 0; irc < this->hsk->get_pv()->nloc; irc++)
    {
        hk[irc] += h_lambda[irc];
    }
    //std::cout << "OperatorScLambda contributeHk" << std::endl;
    ModuleBase::timer::tick("OperatorScLambda", "contributeHk");
}

// contribute to Hk
template <>
void OperatorScLambda<OperatorLCAO<std::complex<double>, double>>::contributeHk(int ik)
{
    ModuleBase::TITLE("OperatorScLambda", "contributeHk");
    ModuleBase::timer::tick("OperatorScLambda", "contributeHk");
    SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>& sc
        = SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::getScInstance();
    std::vector<std::complex<double>> h_lambda(this->hsk->get_pv()->nloc);
    std::fill(h_lambda.begin(), h_lambda.end(), std::complex<double>(0, 0));
    sc.cal_h_lambda(&h_lambda[0], this->hsk->get_sk(), ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER(PARAM.inp.ks_solver), this->isk[ik]);
    std::complex<double>* hk = this->hsk->get_hk();
    for (int irc = 0; irc < this->hsk->get_pv()->nloc; irc++)
    {
        hk[irc] += h_lambda[irc];
    }
    //std::cout << "OperatorScLambda contributeHk" << std::endl;
    ModuleBase::timer::tick("OperatorScLambda", "contributeHk");
}

// contribute to Hk
template <>
void OperatorScLambda<OperatorLCAO<double, double>>::contributeHk(int ik)
{
    // no need to implement this function
}

template class OperatorScLambda<OperatorLCAO<double, double>>;
template class OperatorScLambda<OperatorLCAO<std::complex<double>, double>>;
template class OperatorScLambda<OperatorLCAO<std::complex<double>, std::complex<double>>>;

} // namespace hamilt
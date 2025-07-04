//
// For EXX in PW.
//
#include "source_psi/psi.h"
#include "source_base/matrix.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "source_pw/hamilt_pwdft/operator_pw/op_exx_pw.h"

#ifndef EXX_HELPER_H
#define EXX_HELPER_H
template <typename T, typename Device>
struct Exx_Helper
{
    using Real = typename GetTypeReal<T>::type;
    using OperatorEXX = hamilt::OperatorEXXPW<T, Device>;

  public:
    Exx_Helper() = default;
    OperatorEXX *op_exx = nullptr;

    void set_firstiter(bool flag = true) { first_iter = flag; }
    void set_wg(const ModuleBase::matrix *wg_) { wg = wg_; }
    void set_psi(psi::Psi<T, Device> *psi_);

    void set_op()
    {
        op_exx->first_iter = first_iter;
        set_psi(psi);
        op_exx->set_wg(wg);
    }

    bool exx_after_converge(int &iter);

    double cal_exx_energy(psi::Psi<T, Device> *psi_);

  private:
    bool first_iter = false;
    psi::Psi<T, Device> *psi = nullptr;
    const ModuleBase::matrix *wg = nullptr;

};
#endif // EXX_HELPER_H

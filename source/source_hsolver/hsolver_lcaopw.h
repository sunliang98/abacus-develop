#ifndef HSOLVERLIP_H
#define HSOLVERLIP_H

#include "source_estate/elecstate.h"
#include "source_hamilt/hamilt.h"
#include "source_base/macros.h"
#include "source_base/module_device/types.h"
namespace hsolver
{

// LCAO-in-PW does not support GPU now.
template <typename T>
class HSolverLIP
{
  private:
    // Note GetTypeReal<T>::type will
    // return T if T is real type(float, double),
    // otherwise return the real type of T(complex<float>, complex<double>)
    using Real = typename GetTypeReal<T>::type;

  public:
    HSolverLIP(ModulePW::PW_Basis_K* wfc_basis_in) : wfc_basis(wfc_basis_in) {};

    /// @brief solve function for lcao_in_pw
    /// @param pHamilt interface to hamilt
    /// @param psi reference to psi
    /// @param pes interface to elecstate
    /// @param transform transformation matrix between lcao and pw
    /// @param skip_charge
    void solve(hamilt::Hamilt<T>* pHamilt,
               psi::Psi<T>& psi,
               elecstate::ElecState* pes,
               psi::Psi<T>& transform,
               const bool skip_charge,
               const double tpiba,
               const int nat);

  private:
    ModulePW::PW_Basis_K* wfc_basis;
};

} // namespace hsolver

#endif
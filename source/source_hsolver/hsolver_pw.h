#ifndef HSOLVERPW_H
#define HSOLVERPW_H

#include "source_estate/elecstate.h"
#include "source_hamilt/hamilt.h"
#include "source_base/macros.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include <unordered_map>
#include "source_base/memory.h"

namespace hsolver
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class HSolverPW
{
  protected:
    // Note GetTypeReal<T>::type will
    // return T if T is real type(float, double),
    // otherwise return the real type of T(complex<float>, complex<double>)
    using Real = typename GetTypeReal<T>::type;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;

  public:
    HSolverPW(ModulePW::PW_Basis_K* wfc_basis_in,
              const std::string calculation_type_in,
              const std::string basis_type_in,
              const std::string method_in,
              const bool use_paw_in,
              const bool use_uspp_in,
              const int nspin_in,
              const int scf_iter_in,
              const int diag_iter_max_in,
              const double diag_thr_in,
              const bool need_subspace_in,
              const bool use_k_continuity_in = false)
        : wfc_basis(wfc_basis_in), calculation_type(calculation_type_in), basis_type(basis_type_in), method(method_in),
          use_paw(use_paw_in), use_uspp(use_uspp_in), nspin(nspin_in), scf_iter(scf_iter_in),
          diag_iter_max(diag_iter_max_in), diag_thr(diag_thr_in), need_subspace(need_subspace_in),
          use_k_continuity(use_k_continuity_in) {};

    /// @brief solve function for pw
    /// @param pHamilt interface to hamilt
    /// @param psi reference to psi
    /// @param pes interface to elecstate
    /// @param method_in dav or cg
    /// @param skip_charge
    void solve(hamilt::Hamilt<T, Device>* pHamilt,
               psi::Psi<T, Device>& psi,
               elecstate::ElecState* pes,
               double* out_eigenvalues,
               const int rank_in_pool_in,
               const int nproc_in_pool_in,
               const bool skip_charge,
               const double tpiba,
               const int nat);


  protected:
    // diago caller
    void hamiltSolvePsiK(hamilt::Hamilt<T, Device>* hm,
                         psi::Psi<T, Device>& psi,
                         std::vector<Real>& pre_condition,
                         Real* eigenvalue,
                         const int& nk_nums);

    // calculate the precondition array for diagonalization in PW base
    void update_precondition(std::vector<Real>& h_diag, const int ik, const int npw, const Real vl_of_0);

    void output_iterInfo();

    ModulePW::PW_Basis_K* wfc_basis;

    const std::string calculation_type;
    const std::string basis_type;
    const std::string method;
    const bool use_paw;
    const bool use_uspp;
    const int nspin;

    const int scf_iter;      // Start from 1
    const int diag_iter_max; // max iter times for diagonalization
    const double diag_thr;   // threshold for diagonalization

    const bool need_subspace; // for cg or dav_subspace

    const bool use_k_continuity;

  protected:
    Device* ctx = {};

    int rank_in_pool = 0;
    int nproc_in_pool = 1;

    std::vector<double> ethr_band;

  private:
    /// @brief calculate the threshold for iterative-diagonalization for each band
    void cal_smooth_ethr(const double& wk, const double* wg, const double& ethr, std::vector<double>& ethrs);

#ifdef USE_PAW
    void paw_func_in_kloop(const int ik,
                           const double tpiba);

    void call_paw_cell_set_currentk(const int ik);

    void paw_func_after_kloop(psi::Psi<T, Device>& psi, elecstate::ElecState* pes,const double tpiba,const int nat);
#endif

    // K-point continuity related members
    std::vector<int> k_order;
    std::unordered_map<int, int> k_parent;
    std::vector<ModuleBase::Vector3<double>> kvecs_c;
    
    void build_k_neighbors();
    void propagate_psi(psi::Psi<T, Device>& psi, const int from_ik, const int to_ik);
};

} // namespace hsolver

#endif

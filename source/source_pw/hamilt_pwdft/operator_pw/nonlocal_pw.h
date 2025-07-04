#ifndef NONLOCALPW_H
#define NONLOCALPW_H

#include "operator_pw.h"

#include "source_cell/unitcell.h"
#include "source_pw/hamilt_pwdft/kernels/nonlocal_op.h"
#include "source_base/kernels/math_kernel_op.h"

#include "source_pw/hamilt_pwdft/VNL_in_pw.h"

namespace hamilt {

#ifndef NONLOCALTEMPLATE_H
#define NONLOCALTEMPLATE_H

template<class T> class Nonlocal : public T {};
// template<typename Real, typename Device = base_device::DEVICE_CPU>
// class Nonlocal : public OperatorPW<T, Device> {};

#endif

template<typename T, typename Device>
class Nonlocal<OperatorPW<T, Device>> : public OperatorPW<T, Device>
{
  private:
    using Real = typename GetTypeReal<T>::type;
  public:
    Nonlocal(const int* isk_in,
             const pseudopot_cell_vnl* ppcell_in,
             const UnitCell* ucell_in,
             const ModulePW::PW_Basis_K* wfc_basis);

    template<typename T_in, typename Device_in = Device>
    explicit Nonlocal(const Nonlocal<OperatorPW<T_in, Device_in>>* nonlocal);

    virtual ~Nonlocal();

    virtual void init(const int ik_in)override;

    virtual void act(const int nbands,
        const int nbasis,
        const int npol,
        const T* tmpsi_in,
        T* tmhpsi,
        const int ngk_ik = 0,
        const bool is_first_node = false)const override;

    const int *get_isk() const {return this->isk;}
    const pseudopot_cell_vnl *get_ppcell() const {return this->ppcell;}
    const UnitCell *get_ucell() const {return this->ucell;}
    T* get_vkb() const
    {
        return this->vkb;
    }
    T* get_becp() const
    {
        return this->becp;
    }

  private:
    void add_nonlocal_pp(T *hpsi_in, const T *becp, const int m) const;

    mutable int max_npw = 0;

    mutable int npw = 0;

    mutable int npol = 0;

    mutable size_t nkb_m = 0;

    const int* isk = nullptr;

    const pseudopot_cell_vnl* ppcell = nullptr;

    const UnitCell* ucell = nullptr;

    const ModulePW::PW_Basis_K* wfcpw = nullptr;

    mutable T *ps = nullptr;
    mutable T *vkb = nullptr;
    mutable T *becp = nullptr;
    Device* ctx = {};
    base_device::DEVICE_CPU* cpu_ctx = {};
    Real * deeq = nullptr;
    T * deeq_nc = nullptr;
    // using nonlocal_op = nonlocal_pw_op<Real, Device>;
    using gemv_op = ModuleBase::gemv_op<T, Device>;
    using gemm_op = ModuleBase::gemm_op<T, Device>;
    using nonlocal_op = nonlocal_pw_op<Real, Device>;
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    #ifdef __DSP
    using resmem_complex_op = base_device::memory::resize_memory_op_mt<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op_mt<T, Device>;
    #else
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    #endif
    using syncmem_complex_h2d_op = base_device::memory::synchronize_memory_op<T, Device, base_device::DEVICE_CPU>;

    T one{1, 0};
    T zero{0, 0};
};

} // namespace hamilt

#endif
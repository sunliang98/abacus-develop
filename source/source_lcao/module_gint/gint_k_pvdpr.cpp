#include "gint_k.h"
#include "grid_technique.h"
#include "module_parameter/parameter.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/memory.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/tool_threading.h"
#include "source_base/ylm.h"
#include "source_basis/module_ao/ORB_read.h"
#include "source_pw/hamilt_pwdft/global.h"

void Gint_k::allocate_pvdpR(void)
{
    ModuleBase::TITLE("Gint_k","allocate_pvpR");

    const int nspin = PARAM.inp.nspin;
    assert(nspin>0);

    //xiaohui modify 2015-05-30
    // the number of matrix element <phi_0 | V | dphi_R> is this->gridt->nnrg.
    for(int is =0;is<nspin;is++)
    {
        this->pvdpRx_reduced.push_back(hamilt::HContainer<double>(this->ucell->nat));
        pvdpRx_reduced[is].insert_ijrs(this->gridt->get_ijr_info(), *this->ucell);
        pvdpRx_reduced[is].allocate(nullptr, true);
        this->pvdpRy_reduced.push_back(hamilt::HContainer<double>(this->ucell->nat));
        pvdpRy_reduced[is].insert_ijrs(this->gridt->get_ijr_info(), *this->ucell);
        pvdpRy_reduced[is].allocate(nullptr, true);
        this->pvdpRz_reduced.push_back(hamilt::HContainer<double>(this->ucell->nat));
        pvdpRz_reduced[is].insert_ijrs(this->gridt->get_ijr_info(), *this->ucell);
        pvdpRz_reduced[is].allocate(nullptr, true);
    }

    ModuleBase::Memory::record("pvdpR_reduced", 3 * sizeof(double) * this->gridt->nnrg * nspin);
    return;
}

void Gint_k::destroy_pvdpR(void)
{
    ModuleBase::TITLE("Gint_k","destroy_pvpR");

    const int nspin = PARAM.inp.nspin;
    assert(nspin>0);
    pvdpRx_reduced.clear();
    pvdpRy_reduced.clear();
    pvdpRz_reduced.clear();
    pvdpRx_reduced.shrink_to_fit();
    pvdpRy_reduced.shrink_to_fit();
    pvdpRz_reduced.shrink_to_fit();
    return;
}

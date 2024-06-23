#include "diago_iter_assist.h"

#include "module_base/blas_connector.h"
#include "module_base/complexmatrix.h"
#include "module_base/constants.h"
#include "module_base/global_variable.h"
#include "module_base/lapack_connector.h"
#include "module_base/timer.h"
#include "module_base/parallel_reduce.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_hsolver/kernels/dngvd_op.h"
#include "module_base/module_device/device.h"

namespace hsolver{

//----------------------------------------------------------------------
// Hamiltonian diagonalization in the subspace spanned
// by nstart states psi (atomic or random wavefunctions).
// Produces on output n_band eigenvectors (n_band <= nstart) in evc.
//----------------------------------------------------------------------
template<typename T, typename Device>
void DiagoIterAssist<T, Device>::diagH_subspace(
    hamilt::Hamilt<T, Device>* pHamilt, // hamiltonian operator carrier
    const psi::Psi<T, Device> &psi, // [in] wavefunction
    psi::Psi<T, Device> &evc, // [out] wavefunction
    Real *en, // [out] eigenvalues
    int n_band // [in] number of bands to be calculated, also number of rows of evc, if set to 0, n_band = nstart, default 0
    )
{
    ModuleBase::TITLE("DiagoIterAssist", "diagH_subspace");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");

    // two case:
    // 1. pw base: nstart = n_band, psi(nbands * npwx)
    // 2. lcao_in_pw base: nstart >= n_band, psi(NLOCAL * npwx)
    const int nstart = psi.get_nbands();
    if (n_band == 0)
        n_band = nstart;
    assert(n_band <= nstart);

    T* hcc = nullptr, * scc = nullptr, * vcc = nullptr;
    resmem_complex_op()(ctx, hcc, nstart * nstart, "DiagSub::hcc");
    resmem_complex_op()(ctx, scc, nstart * nstart, "DiagSub::scc");
    resmem_complex_op()(ctx, vcc, nstart * nstart, "DiagSub::vcc");
    setmem_complex_op()(ctx, hcc, 0, nstart * nstart);
    setmem_complex_op()(ctx, scc, 0, nstart * nstart);
    setmem_complex_op()(ctx, vcc, 0, nstart * nstart);

    const int dmin = psi.get_current_nbas();
    const int dmax = psi.get_nbasis();

    // qianrui improve this part 2021-3-14
    const T* ppsi = psi.get_pointer();

    // allocated hpsi
    // std::vector<T> hpsi(psi.get_nbands() * psi.get_nbasis());
    T* hphi = nullptr;
    resmem_complex_op()(ctx, hphi, psi.get_nbands() * psi.get_nbasis(), "DiagSub::hpsi");
    setmem_complex_op()(ctx, hphi, 0, psi.get_nbands() * psi.get_nbasis());
    // do hPsi for all bands
    psi::Range all_bands_range(1, psi.get_current_k(), 0, psi.get_nbands()-1);
    hpsi_info hpsi_in(&psi, all_bands_range, hphi);
    pHamilt->ops->hPsi(hpsi_in);

    gemm_op<T, Device>()(
        ctx,
        'C',
        'N',
        nstart,
        nstart,
        dmin,
        &one,
        ppsi,
        dmax,
        hphi,
        dmax,
        &zero,
        hcc,
        nstart
    );
    delmem_complex_op()(ctx, hphi);

    // allocated spsi
    T* sphi = nullptr;
    resmem_complex_op()(ctx, sphi, nstart * dmax, "DiagSub::spsi");
    setmem_complex_op()(ctx, sphi, 0, nstart * dmax);
    // do sPsi for all bands
    pHamilt->sPsi(ppsi, sphi, dmax, dmin, nstart);

    gemm_op<T, Device>()(ctx, 'C', 'N', nstart, nstart, dmin, &one, ppsi, dmax, sphi, dmax, &zero, scc, nstart);
    delmem_complex_op()(ctx, sphi);

    if (GlobalV::NPROC_IN_POOL > 1)
    {
        Parallel_Reduce::reduce_pool(hcc, nstart * nstart);
        Parallel_Reduce::reduce_pool(scc, nstart * nstart);
    }

    // after generation of H and S matrix, diag them
    DiagoIterAssist::diagH_LAPACK(nstart, n_band, hcc, scc, nstart, en, vcc);

    //=======================
    // diagonize the H-matrix
    //=======================
    if (
        (
            (GlobalV::BASIS_TYPE == "lcao")
          ||(GlobalV::BASIS_TYPE == "lcao_in_pw")
        )
      &&(GlobalV::CALCULATION == "nscf")
      )
    {
        GlobalV::ofs_running << " Not do zgemm to get evc." << std::endl;
    }
    else if (
        (
            (GlobalV::BASIS_TYPE == "lcao")
          ||(GlobalV::BASIS_TYPE == "lcao_in_pw")
        )
      &&(
            (GlobalV::CALCULATION == "scf")
          ||(GlobalV::CALCULATION == "md")
          ||(GlobalV::CALCULATION == "relax")
        )
        ) // pengfei 2014-10-13
    {
        // because psi and evc are different here,
        // I think if psi and evc are the same,
        // there may be problems, mohan 2011-01-01
        gemm_op<T, Device>()(
            ctx,
            'N',
            'N',
            dmax,
            n_band,
            nstart,
            &one,
            ppsi, // dmax * nstart
            dmax,
            vcc,  // nstart * n_band
            nstart,
            &zero,
            evc.get_pointer(),
            dmax
        );
    }
    else
    {
        // As the evc and psi may refer to the same matrix, we first
        // create a temporary matrix to store the result. (by wangjp)
        // qianrui improve this part 2021-3-13
        T* evctemp = nullptr;
        resmem_complex_op()(ctx, evctemp, n_band * dmin, "DiagSub::evctemp");
        setmem_complex_op()(ctx, evctemp, 0, n_band * dmin);

        gemm_op<T, Device>()(
            ctx,
            'N',
            'N',
            dmin,
            n_band,
            nstart,
            &one,
            ppsi, // dmin * nstart
            dmax,
            vcc,  // nstart * n_band
            nstart,
            &zero,
            evctemp,
            dmin
        );

        matrixSetToAnother<T, Device>()(ctx, n_band, evctemp, dmin, evc.get_pointer(), dmax);
        // for (int ib = 0; ib < n_band; ib++)
        // {
        //     for (int ig = 0; ig < dmin; ig++)
        //     {
                // evc(ib, ig) = evctmp(ib, ig);
        //     }
        // }
        delmem_complex_op()(ctx, evctemp);
    }

    delmem_complex_op()(ctx, hcc);
    delmem_complex_op()(ctx, scc);
    delmem_complex_op()(ctx, vcc);

    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");
}

template<typename T, typename Device>
void DiagoIterAssist<T, Device>::diagH_subspace_init(
    hamilt::Hamilt<T, Device>* pHamilt,
    const T* psi,
    int psi_nr,
    int psi_nc,
    psi::Psi<T, Device> &evc,
    Real *en)
{
    ModuleBase::TITLE("DiagoIterAssist", "diagH_subspace_init");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace_init");

    // two case:
    // 1. pw base: nstart = n_band, psi(nbands * npwx)
    // 2. lcao_in_pw base: nstart >= n_band, psi(NLOCAL * npwx)

    const int nstart = psi_nr;
    const int n_band = evc.get_nbands();

    Device* ctx = {};

    // ModuleBase::ComplexMatrix hc(nstart, nstart);
    // ModuleBase::ComplexMatrix sc(nstart, nstart);
    // ModuleBase::ComplexMatrix hvec(nstart, n_band);
    T* hcc = nullptr, * scc = nullptr, * vcc = nullptr;
    resmem_complex_op()(ctx, hcc, nstart * nstart, "DiagSub::hcc");
    resmem_complex_op()(ctx, scc, nstart * nstart, "DiagSub::scc");
    resmem_complex_op()(ctx, vcc, nstart * nstart, "DiagSub::vcc");
    setmem_complex_op()(ctx, hcc, 0, nstart * nstart);
    setmem_complex_op()(ctx, scc, 0, nstart * nstart);
    setmem_complex_op()(ctx, vcc, 0, nstart * nstart);

    const int dmin = evc.get_current_nbas();
    const int dmax = evc.get_nbasis();

    // qianrui improve this part 2021-3-14
    // std::complex<double> *aux = new std::complex<double>[dmax * nstart];
    // const std::complex<double> *paux = aux;
    psi::Psi<T, Device> psi_temp(1, nstart, psi_nc, &evc.get_ngk(0));
    syncmem_complex_op ()(ctx, ctx, psi_temp.get_pointer(), psi, psi_temp.size());
    // ModuleBase::GlobalFunc::COPYARRAY(psi, psi_temp.get_pointer(), psi_temp.size());

    const T *ppsi = psi_temp.get_pointer();

    // allocated hpsi
    T* hpsi = nullptr;
    resmem_complex_op()(ctx, hpsi, psi_temp.get_nbands() * psi_temp.get_nbasis(), "DiagSub::hpsi");
    setmem_complex_op()(ctx, hpsi, 0, psi_temp.get_nbands() * psi_temp.get_nbasis());
    // ================================================
    // std::vector<T> hpsi(psi_temp.get_nbands() * psi_temp.get_nbasis());

    // do hPsi for all bands
    if (base_device::get_device_type(ctx) == base_device::GpuDevice)
    {
        for (int i = 0; i < psi_temp.get_nbands(); i++)
        {
            psi::Range band_by_band_range(1, psi_temp.get_current_k(), i, i);
            hpsi_info hpsi_in(&psi_temp, band_by_band_range, hpsi + i * psi_temp.get_nbasis());
            if(pHamilt->ops == nullptr)
            {
                ModuleBase::WARNING("DiagoIterAssist::diagH_subspace_init",
                "Severe warning: Operators in Hamilt are not allocated yet, will return value of psi to evc directly\n");
                for(int iband = 0; iband < evc.get_nbands(); iband++)
                {
                    for(int ig = 0; ig < evc.get_nbasis(); ig++)
                    {
                        evc(iband, ig) = psi[iband * evc.get_nbasis() + ig];
                    }
                    en[iband] = 0.0;
                }
                return;
            }
            pHamilt->ops->hPsi(hpsi_in);
        }
    }
    else if (base_device::get_device_type(ctx) == base_device::CpuDevice)
    {
        psi::Range all_bands_range(1, psi_temp.get_current_k(), 0, psi_temp.get_nbands()-1);
        hpsi_info hpsi_in(&psi_temp, all_bands_range, hpsi);
        if(pHamilt->ops == nullptr)
        {
            ModuleBase::WARNING("DiagoIterAssist::diagH_subspace_init",
            "Severe warning: Operators in Hamilt are not allocated yet, will return value of psi to evc directly\n");
            for(int iband = 0; iband < evc.get_nbands(); iband++)
            {
                for(int ig = 0; ig < evc.get_nbasis(); ig++)
                {
                    evc(iband, ig) = psi[iband * evc.get_nbasis() + ig];
                }
                en[iband] = 0.0;
            }
            return;
        }
        pHamilt->ops->hPsi(hpsi_in);
    }

    gemm_op<T, Device>()(
        ctx,
        'C',
        'N',
        nstart,
        nstart,
        dmin,
        &one,
        ppsi,
        dmax,
        hpsi,
        dmax,
        &zero,
        hcc,
        nstart
    );
    delmem_complex_op()(ctx, hpsi);

    // allocated spsi
    T* spsi = nullptr;
    resmem_complex_op()(ctx, spsi, psi_temp.get_nbands() * psi_temp.get_nbasis(), "DiagSub::spsi");
    setmem_complex_op()(ctx, spsi, 0, psi_temp.get_nbands() * psi_temp.get_nbasis());
    // do sPsi for all bands
    pHamilt->sPsi(ppsi, spsi, psi_temp.get_nbasis(), psi_temp.get_current_nbas(), psi_temp.get_nbands());

    gemm_op<T, Device>()(ctx, 'C', 'N', nstart, nstart, dmin, &one, ppsi, dmax, spsi, dmax, &zero, scc, nstart);
    delmem_complex_op()(ctx, spsi);

    if (GlobalV::NPROC_IN_POOL > 1)
    {
        Parallel_Reduce::reduce_pool(hcc, nstart * nstart);
        Parallel_Reduce::reduce_pool(scc, nstart * nstart);
    }

    // after generation of H and S matrix, diag them
    ///this part only for test, eigenvector would have different phase caused by micro numerical perturbation
    ///set 8 bit effective accuracy would help for debugging
    /*for(int i=0;i<nstart;i++)
    {
        for(int j=0;j<nstart;j++)
        {
            if(std::norm(hc(i,j))<1e-10) hc(i,j) = ModuleBase::ZERO;
            else hc(i,j) = std::complex<double>(double(int(hc(i,j).real()*100000000))/100000000, 0);
            if(std::norm(sc(i,j))<1e-10) sc(i,j) = ModuleBase::ZERO;
            else sc(i,j) = std::complex<double>(double(int(sc(i,j).real()*100000000))/100000000, 0);
        }
    }*/

    DiagoIterAssist::diagH_LAPACK(nstart, n_band, hcc, scc, nstart, en, vcc);

    //=======================
    // diagonize the H-matrix
    //=======================
    if ((GlobalV::BASIS_TYPE == "lcao" || GlobalV::BASIS_TYPE == "lcao_in_pw") && GlobalV::CALCULATION == "nscf")
    {
        GlobalV::ofs_running << " Not do zgemm to get evc." << std::endl;
    }
    else if (
        (
            GlobalV::BASIS_TYPE == "lcao" 
          ||GlobalV::BASIS_TYPE == "lcao_in_pw"
          ||(GlobalV::BASIS_TYPE == "pw" && GlobalV::psi_initializer)
        )
      &&(
            GlobalV::CALCULATION == "scf"
          ||GlobalV::CALCULATION == "md"
          ||GlobalV::CALCULATION == "relax"
        )
        ) // pengfei 2014-10-13
    {
        // because psi and evc are different here,
        // I think if psi and evc are the same,
        // there may be problems, mohan 2011-01-01
        gemm_op<T, Device>()(
            ctx,
            'N',
            'N',
            dmax,
            n_band,
            nstart,
            &one,
            ppsi, // dmax * nstart
            dmax,
            vcc,  // nstart * n_band
            nstart,
            &zero,
            evc.get_pointer(),
            dmax
        );
    }
    else
    {
        // As the evc and psi may refer to the same matrix, we first
        // create a temporary matrix to store the result. (by wangjp)
        // qianrui improve this part 2021-3-13
        T* evctemp = nullptr;
        resmem_complex_op()(ctx, evctemp, n_band * dmin, "DiagSub::evctemp");
        setmem_complex_op()(ctx, evctemp, 0, n_band * dmin);

        gemm_op<T, Device>()(
            ctx,
            'N',
            'N',
            dmin,
            n_band,
            nstart,
            &one,
            ppsi, // dmin * nstart
            dmax,
            vcc,  // nstart * n_band
            nstart,
            &zero,
            evctemp,
            dmin
        );

        matrixSetToAnother<T, Device>()(ctx, n_band, evctemp, dmin, evc.get_pointer(), dmax);

        delmem_complex_op()(ctx, evctemp);
    }

    delmem_complex_op()(ctx, hcc);
    delmem_complex_op()(ctx, scc);
    delmem_complex_op()(ctx, vcc);
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace_init");
}

template<typename T, typename Device>
void DiagoIterAssist<T, Device>::diagH_LAPACK(
    const int nstart,
    const int nbands,
    const T* hcc,
    const T* scc,
    const int ldh, // nstart
    Real *e, // always in CPU
    T* vcc)
{
    ModuleBase::TITLE("DiagoIterAssist", "diagH_LAPACK");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_LAPACK");

    Real* eigenvalues = nullptr;
    resmem_var_op()(ctx, eigenvalues, nstart);
    setmem_var_op()(ctx, eigenvalues, 0, nstart);

    dngvd_op<T, Device>()(ctx, nstart, ldh, hcc, scc, eigenvalues, vcc);

    if (base_device::get_device_type<Device>(ctx) == base_device::GpuDevice)
    {
#if ((defined __CUDA) || (defined __ROCM))
        // set eigenvalues in GPU to e in CPU
        syncmem_var_d2h_op()(cpu_ctx, gpu_ctx, e, eigenvalues, nbands);
#endif
    }
    else if (base_device::get_device_type<Device>(ctx) == base_device::CpuDevice)
    {
        // set eigenvalues in CPU to e in CPU
        syncmem_var_op()(ctx, ctx, e, eigenvalues, nbands);
    }

    delmem_var_op()(ctx, eigenvalues);


    // const bool all_eigenvalues = (nstart == nbands);
    // if (all_eigenvalues) {
    //     //===========================
    //     // calculate all eigenvalues
    //     //===========================
    //     // dngv_op<Real, Device>()(ctx, nstart, ldh, hcc, scc, res, vcc);
    //     dngvd_op<Real, Device>()(ctx, nstart, ldh, hcc, scc, res, vcc);
    // }
    // else {
    //     //=====================================
    //     // calculate only m lowest eigenvalues
    //     //=====================================
    //     dngvx_op<Real, Device>()(ctx, nstart, ldh, hcc, scc, nbands, res, vcc);
    // }

    ModuleBase::timer::tick("DiagoIterAssist", "diagH_LAPACK");
}

template<typename T, typename Device>
bool DiagoIterAssist<T, Device>::test_exit_cond(const int &ntry, const int &notconv)
{
    //================================================================
    // If this logical function is true, need to do diagH_subspace
    // and cg again.
    //================================================================

    bool scf = true;
    if (GlobalV::CALCULATION == "nscf")
        scf = false;

    // If ntry <=5, try to do it better, if ntry > 5, exit.
    const bool f1 = (ntry <= 5);

    // In non-self consistent calculation, do until totally converged.
    const bool f2 = ((!scf && (notconv > 0)));

    // if self consistent calculation, if not converged > 5,
    // using diagH_subspace and cg method again. ntry++
    const bool f3 = ((scf && (notconv > 5)));
    return (f1 && (f2 || f3));
}

template class DiagoIterAssist<std::complex<float>, base_device::DEVICE_CPU>;
template class DiagoIterAssist<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoIterAssist<std::complex<float>, base_device::DEVICE_GPU>;
template class DiagoIterAssist<std::complex<double>, base_device::DEVICE_GPU>;
#endif

#ifdef __LCAO
template class DiagoIterAssist<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoIterAssist<double, base_device::DEVICE_GPU>;
#endif
#endif
} // namespace hsolver

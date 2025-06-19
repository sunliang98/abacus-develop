#ifdef __MPI
#include "dm_trans.h"
#include "source_base/scalapack_connector.h"
#include "source_base/tool_title.h"
#include "module_lr/utils/lr_util.h"
namespace LR
{

    //output: col first, consistent with blas
    // c: nao*nbands in para2d, nbands*nao in psi  (row-para and constructed: nao)
    // X: nvirt*nocc in para2d, nocc*nvirt in psi (row-para and constructed: nvirt)
template <>
std::vector<container::Tensor> cal_dm_trans_pblas(const double* const X_istate,
    const Parallel_2D& px,
    const psi::Psi<double>& c,
    const Parallel_2D& pc,
    const int naos,
    const int nocc,
    const int nvirt,
    const Parallel_2D& pmat,
    const double factor,
    const MO_TYPE type)
{
    ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_pblas");
    assert(px.comm() == pc.comm() && px.comm() == pmat.comm());
    assert(px.blacs_ctxt == pc.blacs_ctxt && px.blacs_ctxt == pmat.blacs_ctxt);
    assert(pmat.get_local_size() > 0);

    const int nks = c.get_nk();
    const int i1 = 1;
    const int ivirt = nocc + 1;
    const int nmo1 = type == MO_TYPE::VV ? nvirt : nocc;
    const int nmo2 = type == MO_TYPE::OO ? nocc : nvirt;
    const int imo1 = type == MO_TYPE::VV ? ivirt : i1;
    const int imo2 = type == MO_TYPE::OO ? i1 : ivirt;

    std::vector<container::Tensor> dm_trans(nks,
        container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { pmat.get_col_size(), pmat.get_row_size() }));
    for (int isk = 0; isk < nks; ++isk)
    {
        c.fix_k(isk);
        const int x_start = isk * px.get_local_size();

        char transa = 'N';
        char transb = 'T';
        const double alpha = 1.0;
        const double beta = 0;

        // 1. [X*C_occ^T]^T=C_occ*X^T
        Parallel_2D pXc; // nvirt*naos
        LR_Util::setup_2d_division(pXc, px.get_block_size(), naos, nmo2, px.blacs_ctxt);
        container::Tensor Xc(DAT::DT_DOUBLE,
                             DEV::CpuDevice,
                             {pXc.get_col_size(), pXc.get_row_size()}); // row is "inside"(memory contiguity) for pblas
        Xc.zero();
        pdgemm_(&transa, &transb, &naos, &nmo2, &nmo1,
            &alpha, c.get_pointer(), &i1, &imo1, pc.desc,
            X_istate + x_start, &i1, &i1, px.desc,
            &beta, Xc.data<double>(), &i1, &i1, pXc.desc);

        // 2. C_virt*[X*C_occ^T]
        pdgemm_(&transa, &transb, &naos, &naos, &nmo2,
            &factor, c.get_pointer(), &i1, &imo2, pc.desc,
            Xc.data<double>(), &i1, &i1, pXc.desc,
            &beta, dm_trans[isk].data<double>(), &i1, &i1, pmat.desc);
    }
    return dm_trans;
}
template <>
std::vector<container::Tensor> cal_dm_trans_pblas(const std::complex<double>* const X_istate,
    const Parallel_2D& px,
    const psi::Psi<std::complex<double>>& c,
    const Parallel_2D& pc,
    const int naos,
    const int nocc,
    const int nvirt,
    const Parallel_2D& pmat,
    const std::complex<double> factor,
    const MO_TYPE type)
{
    ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_pblas");
    assert(px.comm() == pc.comm() && px.comm() == pmat.comm());
    assert(px.blacs_ctxt == pc.blacs_ctxt && px.blacs_ctxt == pmat.blacs_ctxt);
    assert(pmat.get_local_size() > 0);
    const int nks = c.get_nk();
    const int i1 = 1;
    const int ivirt = nocc + 1;
    const int nmo1 = type == MO_TYPE::VV ? nvirt : nocc;
    const int nmo2 = type == MO_TYPE::OO ? nocc : nvirt;
    const int imo1 = type == MO_TYPE::VV ? ivirt : i1;
    const int imo2 = type == MO_TYPE::OO ? i1 : ivirt;

    std::vector<container::Tensor> dm_trans(nks,
        container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, {pmat.get_col_size(), pmat.get_row_size()}));
    for (int isk = 0; isk < nks; ++isk)
    {
        c.fix_k(isk);
        const int x_start = isk * px.get_local_size();

        // ============== C_virt * X * C_occ^\dagger=============
        // char transa = 'N';
        // char transb = 'C';
        // // 1. [X*C_occ^\dagger]^\dagger=C_occ*X^\dagger
        // Parallel_2D pXc;
        // LR_Util::setup_2d_division(pXc, px.get_block_size(), naos, nvirt, px.comm_2D, px.blacs_ctxt);
        // container::Tensor Xc(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { pXc.get_col_size(), pXc.get_row_size()
        // });//row is "inside"(memory contiguity) for pblas Xc.zero(); const std::complex<double> alpha(1.0, 0.0);
        // const std::complex<double> beta(0.0, 0.0);
        // pzgemm_(&transa, &transb, &naos, &nvirt, &nocc,
        //     &alpha, c.get_pointer(), &i1, &i1, pc.desc,
        //     X_istate + x_start, &i1, &i1, px.desc,
        //     &beta, Xc.data<std::complex<double>>(), &i1, &i1, pXc.desc);

        // // 2. C_virt*[X*C_occ^\dagger]
        // pzgemm_(&transa, &transb, &naos, &naos, &nvirt,
        //     &alpha, c.get_pointer(), &i1, &ivirt, pc.desc,
        //     Xc.data<std::complex<double>>(), &i1, &i1, pXc.desc,
        //     &beta, dm_trans[isk].data<std::complex<double>>(), &i1, &i1, pmat.desc);

        // ============== [C_virt * X * C_occ^\dagger]^T=============
        // ============== = [C_occ^* * X^T * C_virt^T]=============
        // 1. X*C_occ^\dagger
        char transa = 'N';
        char transb = 'C';
        Parallel_2D pXc;
        LR_Util::setup_2d_division(pXc, px.get_block_size(), nmo2, naos, px.blacs_ctxt);
        container::Tensor Xc(DAT::DT_COMPLEX_DOUBLE,
                             DEV::CpuDevice,
                             {pXc.get_col_size(), pXc.get_row_size()}); // row is "inside"(memory contiguity) for pblas
        Xc.zero();
        const std::complex<double> alpha(1.0, 0.0);
        const std::complex<double> beta(0.0, 0.0);
        pzgemm_(&transa, &transb, &nmo2, &naos, &nmo1, &alpha,
            X_istate + x_start, &i1, &i1, px.desc,
            c.get_pointer(), &i1, &imo1, pc.desc,
            &beta, Xc.data<std::complex<double>>(), &i1, &i1, pXc.desc);

        // 2. [X*C_occ^\dagger]^TC_virt^T
        transa = transb = 'T';
        pzgemm_(&transa, &transb, &naos, &naos, &nmo2,
            &factor, Xc.data<std::complex<double>>(), &i1, &i1, pXc.desc,
            c.get_pointer(), &i1, &imo2, pc.desc,
            &beta, dm_trans[isk].data<std::complex<double>>(), &i1, &i1, pmat.desc);
    }
    return dm_trans;
}
}
#endif

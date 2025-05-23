/// cal_v_delta_precalc : v_delta_precalc is used for training with v_delta label,
//                         which equals gevdm * v_delta_pdm,
//                         v_delta_pdm = overlap * overlap
/// check_v_delta_precalc : check v_delta_precalc
//  prepare_phialpha : prepare phialpha for outputting npy file
//  prepare_gevdm : prepare gevdm for outputting npy file

#ifdef __DEEPKS

#include "deepks_vdpre.h"

#include "LCAO_deepks_io.h" // mohan add 2024-07-22
#include "deepks_iterate.h"
#include "module_base/blas_connector.h"
#include "module_base/constants.h"
#include "module_base/libm/libm.h"
#include "module_base/parallel_reduce.h"
#include "module_hamilt_lcao/module_hcontainer/atom_pair.h"
#include "module_parameter/parameter.h"

// calculates v_delta_precalc[nks,nlocal,nlocal,NAt,NDscrpt] = gevdm * v_delta_pdm;
// v_delta_pdm[nks,nlocal,nlocal,Inl,nm,nm] = overlap * overlap;
// for deepks_v_delta = 1
template <typename TK>
void DeePKS_domain::cal_v_delta_precalc(const int nlocal,
                                        const int lmaxd,
                                        const int inlmax,
                                        const int nat,
                                        const int nks,
                                        const std::vector<int>& inl2l,
                                        const std::vector<ModuleBase::Vector3<double>>& kvec_d,
                                        const std::vector<hamilt::HContainer<double>*> phialpha,
                                        const std::vector<torch::Tensor> gevdm,
                                        const ModuleBase::IntArray* inl_index,
                                        const UnitCell& ucell,
                                        const LCAO_Orbitals& orb,
                                        const Parallel_Orbitals& pv,
                                        const Grid_Driver& GridD,
                                        torch::Tensor& v_delta_precalc)
{
    ModuleBase::TITLE("DeePKS_domain", "cal_v_delta_precalc");
    ModuleBase::timer::tick("DeePKS_domain", "cal_v_delta_precalc");
    // timeval t_start;
    // gettimeofday(&t_start,NULL);

    constexpr torch::Dtype dtype = std::is_same<TK, double>::value ? torch::kFloat64 : torch::kComplexDouble;
    using TK_tensor = typename std::conditional<std::is_same<TK, std::complex<double>>::value, c10::complex<double>, TK>::type;

    torch::Tensor v_delta_pdm
        = torch::zeros({nks, nlocal, nlocal, inlmax, (2 * lmaxd + 1), (2 * lmaxd + 1)}, torch::dtype(dtype));
    auto accessor
        = v_delta_pdm.accessor<std::conditional_t<std::is_same<TK, double>::value, double, c10::complex<double>>, 6>();

    DeePKS_domain::iterate_ad2(
        ucell,
        GridD,
        orb,
        false, // no trace_alpha
        [&](const int iat,
            const ModuleBase::Vector3<double>& tau0,
            const int ibt1,
            const ModuleBase::Vector3<double>& tau1,
            const int start1,
            const int nw1_tot,
            ModuleBase::Vector3<int> dR1,
            const int ibt2,
            const ModuleBase::Vector3<double>& tau2,
            const int start2,
            const int nw2_tot,
            ModuleBase::Vector3<int> dR2)
        {
            const int T0 = ucell.iat2it[iat];
            const int I0 = ucell.iat2ia[iat];
            if (phialpha[0]->find_matrix(iat, ibt1, dR1.x, dR1.y, dR1.z) == nullptr
                || phialpha[0]->find_matrix(iat, ibt2, dR2.x, dR2.y, dR2.z) == nullptr)
            {
                return; // to next loop
            }

            for (int iw1 = 0; iw1 < nw1_tot; ++iw1)
            {
                const int iw1_all = start1 + iw1; // this is \mu
                const int iw1_local = pv.global2local_row(iw1_all);
                if (iw1_local < 0)
                {
                    continue;
                }
                for (int iw2 = 0; iw2 < nw2_tot; ++iw2)
                {
                    const int iw2_all = start2 + iw2; // this is \nu
                    const int iw2_local = pv.global2local_col(iw2_all);
                    if (iw2_local < 0)
                    {
                        continue;
                    }

                    hamilt::BaseMatrix<double>* overlap_1 = phialpha[0]->find_matrix(iat, ibt1, dR1);
                    hamilt::BaseMatrix<double>* overlap_2 = phialpha[0]->find_matrix(iat, ibt2, dR2);
                    assert(overlap_1->get_col_size() == overlap_2->get_col_size());

                    for (int ik = 0; ik < nks; ik++)
                    {
                        int ib = 0;
                        std::complex<double> kphase = std::complex<double>(1.0, 0.0);
                        if (std::is_same<TK, std::complex<double>>::value)
                        {
                            const double arg
                                = -(kvec_d[ik] * ModuleBase::Vector3<double>(dR1 - dR2)) * ModuleBase::TWO_PI;
                            kphase = std::complex<double>(cos(arg), sin(arg));
                        }
                        TK_tensor* kpase_ptr = reinterpret_cast<TK_tensor*>(&kphase);
                        for (int L0 = 0; L0 <= orb.Alpha[0].getLmax(); ++L0)
                        {
                            for (int N0 = 0; N0 < orb.Alpha[0].getNchi(L0); ++N0)
                            {
                                const int inl = inl_index[T0](I0, L0, N0);
                                const int nm = 2 * L0 + 1;
                                for (int m1 = 0; m1 < nm; ++m1) // nm = 1 for s, 3 for p, 5 for d
                                {
                                    for (int m2 = 0; m2 < nm; ++m2) // nm = 1 for s, 3 for p, 5 for d
                                    {
                                        TK_tensor tmp = overlap_1->get_value(iw1, ib + m1) * overlap_2->get_value(iw2, ib + m2) * *kpase_ptr;
                                        accessor[ik][iw1][iw2][inl][m1][m2] += tmp;
                                    }
                                }
                                ib += nm;
                            }
                        }
                    } // ik
                }     // iw2
            }         // iw1
        }
    );
#ifdef __MPI
    const int size = nks * nlocal * nlocal * inlmax * (2 * lmaxd + 1) * (2 * lmaxd + 1);
    TK_tensor* data_tensor_ptr = v_delta_pdm.data_ptr<TK_tensor>();
    TK* data_ptr = reinterpret_cast<TK*>(data_tensor_ptr);
    Parallel_Reduce::reduce_all(data_ptr, size);
#endif

    // transfer v_delta_pdm to v_delta_pdm_vector
    int nlmax = inlmax / nat;

    std::vector<torch::Tensor> v_delta_pdm_vector;
    for (int nl = 0; nl < nlmax; ++nl)
    {
        std::vector<torch::Tensor> kuuammv;
        for (int iks = 0; iks < nks; ++iks)
        {
            std::vector<torch::Tensor> uuammv;
            for (int mu = 0; mu < nlocal; ++mu)
            {
                std::vector<torch::Tensor> uammv;
                for (int nu = 0; nu < nlocal; ++nu)
                {
                    std::vector<torch::Tensor> ammv;
                    for (int iat = 0; iat < nat; ++iat)
                    {
                        int inl = iat * nlmax + nl;
                        int nm = 2 * inl2l[inl] + 1;
                        std::vector<TK> mmv;

                        for (int m1 = 0; m1 < nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                        {
                            for (int m2 = 0; m2 < nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                            {
                                TK_tensor tmp = accessor[iks][mu][nu][inl][m1][m2];
                                TK* tmp_ptr = reinterpret_cast<TK*>(&tmp);
                                mmv.push_back(*tmp_ptr); 
                            }
                        }
                        torch::Tensor mm = torch::from_blob(mmv.data(),
                                                            {nm, nm},
                                                            torch::TensorOptions().dtype(dtype))
                                               .clone(); // nm*nm
                        ammv.push_back(mm);
                    }
                    torch::Tensor amm = torch::stack(ammv, 0);
                    uammv.push_back(amm);
                }
                torch::Tensor uamm = torch::stack(uammv, 0);
                uuammv.push_back(uamm);
            }
            torch::Tensor uuamm = torch::stack(uuammv, 0);
            kuuammv.push_back(uuamm);
        }
        torch::Tensor kuuamm = torch::stack(kuuammv, 0);
        v_delta_pdm_vector.push_back(kuuamm);
    }

    assert(v_delta_pdm_vector.size() == nlmax);

    // einsum for each nl:
    std::vector<torch::Tensor> v_delta_precalc_vector;
    for (int nl = 0; nl < nlmax; ++nl)
    {
        torch::Tensor gevdm_totype = gevdm[nl].to(dtype);
        v_delta_precalc_vector.push_back(at::einsum("kxyamn, avmn->kxyav", {v_delta_pdm_vector[nl], gevdm_totype}));
    }

    v_delta_precalc = torch::cat(v_delta_precalc_vector, -1);
    //  timeval t_end;
    //  gettimeofday(&t_end,NULL);
    //  std::cout<<"calculate v_delta_precalc time:\t"<<(double)(t_end.tv_sec-t_start.tv_sec) +
    //  (double)(t_end.tv_usec-t_start.tv_usec)/1000000.0<<std::endl;

    ModuleBase::timer::tick("DeePKS_domain", "cal_v_delta_precalc");
    return;
}

template <typename TK>
void DeePKS_domain::check_v_delta_precalc(const int nat,
                                          const int nks,
                                          const int nlocal,
                                          const int des_per_atom,
                                          const torch::Tensor& v_delta_precalc)
{
    using TK_tensor = typename std::conditional<std::is_same<TK, std::complex<double>>::value, c10::complex<double>, TK>::type;
    std::ofstream ofs("v_delta_precalc.dat");
    ofs << std::setprecision(10);
    auto accessor
        = v_delta_precalc
              .accessor<std::conditional_t<std::is_same<TK, double>::value, double, c10::complex<double>>, 5>();
    for (int iks = 0; iks < nks; ++iks)
    {
        for (int mu = 0; mu < nlocal; ++mu)
        {
            for (int nu = 0; nu < nlocal; ++nu)
            {
                for (int iat = 0; iat < nat; ++iat)
                {
                    for (int p = 0; p < des_per_atom; ++p)
                    {
                        TK_tensor tmp = accessor[iks][mu][nu][iat][p];
                        TK* tmp_ptr = reinterpret_cast<TK*>(&tmp);
                        ofs << *tmp_ptr << " ";
                    }
                }
                ofs << std::endl;
            }
        }
    }
    ofs.close();
}

// prepare_phialpha and prepare_gevdm for deepks_v_delta = 2
template <typename TK>
void DeePKS_domain::prepare_phialpha(const int nlocal,
                                     const int lmaxd,
                                     const int inlmax,
                                     const int nat,
                                     const int nks,
                                     const std::vector<ModuleBase::Vector3<double>>& kvec_d,
                                     const std::vector<hamilt::HContainer<double>*> phialpha,
                                     const UnitCell& ucell,
                                     const LCAO_Orbitals& orb,
                                     const Parallel_Orbitals& pv,
                                     const Grid_Driver& GridD,
                                     torch::Tensor& phialpha_out)
{
    ModuleBase::TITLE("DeePKS_domain", "prepare_phialpha");
    ModuleBase::timer::tick("DeePKS_domain", "prepare_phialpha");
    constexpr torch::Dtype dtype = std::is_same<TK, double>::value ? torch::kFloat64 : torch::kComplexDouble;
    using TK_tensor = typename std::conditional<std::is_same<TK, std::complex<double>>::value, c10::complex<double>, TK>::type;
    int nlmax = inlmax / nat;
    int mmax = 2 * lmaxd + 1;
    phialpha_out = torch::zeros({nat, nlmax, nks, nlocal, mmax}, dtype);
    auto accessor
        = phialpha_out.accessor<std::conditional_t<std::is_same<TK, double>::value, double, c10::complex<double>>, 5>();

    DeePKS_domain::iterate_ad1(
        ucell,
        GridD,
        orb,
        false, // no trace_alpha
        [&](const int iat,
            const ModuleBase::Vector3<double>& tau0,
            const int ibt,
            const ModuleBase::Vector3<double>& tau,
            const int start,
            const int nw_tot,
            ModuleBase::Vector3<int> dR)
        {
            if (phialpha[0]->find_matrix(iat, ibt, dR.x, dR.y, dR.z) == nullptr)
            {
                return; // to next loop
            }

            // middle loop : all atomic basis on the adjacent atom ad
            for (int iw1 = 0; iw1 < nw_tot; ++iw1)
            {
                const int iw1_all = start + iw1;
                const int iw1_local = pv.global2local_row(iw1_all);
                const int iw2_local = pv.global2local_col(iw1_all);
                if (iw1_local < 0 || iw2_local < 0)
                {
                    continue;
                }
                hamilt::BaseMatrix<double>* overlap = phialpha[0]->find_matrix(iat, ibt, dR);

                for (int ik = 0; ik < nks; ik++)
                {
                    std::complex<double> kphase = std::complex<double>(1.0, 0.0);
                    if (std::is_same<TK, std::complex<double>>::value)
                    {
                        const double arg = -(kvec_d[ik] * ModuleBase::Vector3<double>(dR)) * ModuleBase::TWO_PI;
                        kphase = std::complex<double>(cos(arg), sin(arg));
                    }
                    TK_tensor* kpase_ptr = reinterpret_cast<TK_tensor*>(&kphase);
                    int ib = 0;
                    int nl = 0;
                    for (int L0 = 0; L0 <= orb.Alpha[0].getLmax(); ++L0)
                    {
                        for (int N0 = 0; N0 < orb.Alpha[0].getNchi(L0); ++N0)
                        {
                            const int nm = 2 * L0 + 1;
                            for (int m1 = 0; m1 < nm; ++m1) // nm = 1 for s, 3 for p, 5 for d
                            {
                                TK_tensor tmp = overlap->get_value(iw1, ib + m1) * *kpase_ptr;
                                accessor[iat][nl][ik][iw1_local][m1] += tmp;
                            }
                            ib += nm;
                            nl++;
                        }
                    }
                } // end ik
            }     // end iw
        }
    );

#ifdef __MPI
    int size = nat * nlmax * nks * nlocal * mmax;
    TK_tensor* data_tensor_ptr = phialpha_out.data_ptr<TK_tensor>();
    TK* data_ptr = reinterpret_cast<TK*>(data_tensor_ptr);
    Parallel_Reduce::reduce_all(data_ptr, size);

#endif

    ModuleBase::timer::tick("DeePKS_domain", "prepare_phialpha");
    return;
}

template <typename TK>
void DeePKS_domain::check_vdp_phialpha(const int nat,
                                       const int nks,
                                       const int nlocal,
                                       const int inlmax,
                                       const int lmaxd,
                                       const torch::Tensor& phialpha_out)
{
    using TK_tensor = typename std::conditional<std::is_same<TK, std::complex<double>>::value, c10::complex<double>, TK>::type;
    std::ofstream ofs("vdp_phialpha.dat");
    ofs << std::setprecision(10);
    auto accessor
        = phialpha_out.accessor<std::conditional_t<std::is_same<TK, double>::value, double, c10::complex<double>>, 5>();

    int nlmax = inlmax / nat;
    int mmax = 2 * lmaxd + 1;
    for (int iat = 0; iat < nat; iat++)
    {
        for (int nl = 0; nl < nlmax; nl++)
        {
            for (int iks = 0; iks < nks; iks++)
            {
                for (int mu = 0; mu < nlocal; mu++)
                {
                    for (int m = 0; m < mmax; m++)
                    {
                        TK_tensor tmp = accessor[iat][nl][iks][mu][m];
                        TK* tmp_ptr = reinterpret_cast<TK*>(&tmp);
                        ofs << *tmp_ptr << " ";
                    }
                }
            }
            ofs << std::endl;
        }
    }
    ofs.close();
}

void DeePKS_domain::prepare_gevdm(const int nat,
                                  const int lmaxd,
                                  const int inlmax,
                                  const LCAO_Orbitals& orb,
                                  const std::vector<torch::Tensor>& gevdm_in,
                                  torch::Tensor& gevdm_out)
{
    ModuleBase::TITLE("DeePKS_domain", "prepare_gevdm");
    ModuleBase::timer::tick("DeePKS_domain", "prepare_gevdm");
    int nlmax = inlmax / nat;
    int mmax = 2 * lmaxd + 1;
    gevdm_out = torch::zeros({nat, nlmax, mmax, mmax, mmax}, torch::TensorOptions().dtype(torch::kFloat64));

    int nl = 0;
    for (int L0 = 0; L0 <= orb.Alpha[0].getLmax(); ++L0)
    {
        for (int N0 = 0; N0 < orb.Alpha[0].getNchi(L0); ++N0)
        {
            for (int iat = 0; iat < nat; iat++)
            {
                const int nm = 2 * L0 + 1;
                for (int v = 0; v < nm; ++v) // nm = 1 for s, 3 for p, 5 for d
                {
                    for (int m = 0; m < nm; ++m)
                    {
                        for (int n = 0; n < nm; ++n)
                        {
                            gevdm_out[iat][nl][v][m][n] = gevdm_in[nl][iat][v][m][n];
                        }
                    }
                }
            }
            nl++;
        }
    }
    assert(nl == nlmax);

    ModuleBase::timer::tick("DeePKS_domain", "prepare_gevdm");
    return;
}

void DeePKS_domain::check_vdp_gevdm(const int nat, const int lmaxd, const int inlmax, const torch::Tensor& gevdm)
{
    std::ofstream ofs("vdp_gevdm.dat");
    ofs << std::setprecision(10);

    auto accessor = gevdm.accessor<double, 5>();

    int nlmax = inlmax / nat;
    int mmax = 2 * lmaxd + 1;
    for (int iat = 0; iat < nat; iat++)
    {
        for (int nl = 0; nl < nlmax; nl++)
        {
            for (int v = 0; v < mmax; v++)
            {
                for (int m = 0; m < mmax; m++)
                {
                    for (int n = 0; n < mmax; n++)
                    {
                        ofs << accessor[iat][nl][v][m][n] << " ";
                    }
                }
            }
            ofs << std::endl;
        }
    }
    ofs.close();
}

template void DeePKS_domain::cal_v_delta_precalc<double>(const int nlocal,
                                                         const int lmaxd,
                                                         const int inlmax,
                                                         const int nat,
                                                         const int nks,
                                                         const std::vector<int>& inl2l,
                                                         const std::vector<ModuleBase::Vector3<double>>& kvec_d,
                                                         const std::vector<hamilt::HContainer<double>*> phialpha,
                                                         const std::vector<torch::Tensor> gevdm,
                                                         const ModuleBase::IntArray* inl_index,
                                                         const UnitCell& ucell,
                                                         const LCAO_Orbitals& orb,
                                                         const Parallel_Orbitals& pv,
                                                         const Grid_Driver& GridD,
                                                         torch::Tensor& v_delta_precalc);
template void DeePKS_domain::cal_v_delta_precalc<std::complex<double>>(
    const int nlocal,
    const int lmaxd,
    const int inlmax,
    const int nat,
    const int nks,
    const std::vector<int>& inl2l,
    const std::vector<ModuleBase::Vector3<double>>& kvec_d,
    const std::vector<hamilt::HContainer<double>*> phialpha,
    const std::vector<torch::Tensor> gevdm,
    const ModuleBase::IntArray* inl_index,
    const UnitCell& ucell,
    const LCAO_Orbitals& orb,
    const Parallel_Orbitals& pv,
    const Grid_Driver& GridD,
    torch::Tensor& v_delta_precalc);

template void DeePKS_domain::check_v_delta_precalc<double>(const int nat,
                                                           const int nks,
                                                           const int nlocal,
                                                           const int des_per_atom,
                                                           const torch::Tensor& v_delta_precalc);
template void DeePKS_domain::check_v_delta_precalc<std::complex<double>>(const int nat,
                                                                         const int nks,
                                                                         const int nlocal,
                                                                         const int des_per_atom,
                                                                         const torch::Tensor& v_delta_precalc);

template void DeePKS_domain::prepare_phialpha<double>(const int nlocal,
                                                      const int lmaxd,
                                                      const int inlmax,
                                                      const int nat,
                                                      const int nks,
                                                      const std::vector<ModuleBase::Vector3<double>>& kvec_d,
                                                      const std::vector<hamilt::HContainer<double>*> phialpha,
                                                      const UnitCell& ucell,
                                                      const LCAO_Orbitals& orb,
                                                      const Parallel_Orbitals& pv,
                                                      const Grid_Driver& GridD,
                                                      torch::Tensor& phialpha_out);

template void DeePKS_domain::prepare_phialpha<std::complex<double>>(
    const int nlocal,
    const int lmaxd,
    const int inlmax,
    const int nat,
    const int nks,
    const std::vector<ModuleBase::Vector3<double>>& kvec_d,
    const std::vector<hamilt::HContainer<double>*> phialpha,
    const UnitCell& ucell,
    const LCAO_Orbitals& orb,
    const Parallel_Orbitals& pv,
    const Grid_Driver& GridD,
    torch::Tensor& phialpha_out);

template void DeePKS_domain::check_vdp_phialpha<double>(const int nat,
                                                        const int nks,
                                                        const int nlocal,
                                                        const int inlmax,
                                                        const int lmaxd,
                                                        const torch::Tensor& phialpha_out);
template void DeePKS_domain::check_vdp_phialpha<std::complex<double>>(const int nat,
                                                                      const int nks,
                                                                      const int nlocal,
                                                                      const int inlmax,
                                                                      const int lmaxd,
                                                                      const torch::Tensor& phialpha_out);

#endif

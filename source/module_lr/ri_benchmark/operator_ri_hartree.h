#include "module_hamilt_general/operator.h"
#include "module_lr/ri_benchmark/ri_benchmark.h"
#include "module_lr/utils/lr_util_print.h"
namespace RI_Benchmark
{
    template<typename T>
    class OperatorRIHartree : public hamilt::Operator<T>
    {
        using TR = T;
    public:
        OperatorRIHartree(const UnitCell& ucell,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const psi::Psi<T>& psi_ks_in,
            const TLRI<TR>& Cs_ao,
            const TLRI<TR>& Vs, 
            const bool& read_from_aims=false,
            const std::vector<int>& aims_nbasis={})
            : naos(naos), nocc(nocc), nvirt(nvirt), npairs(nocc* nvirt), psi_ks(psi_ks_in),
            Cs_ao(Cs_ao), Vs(Vs),
            Cs_vo_mo(cal_Cs_mo(ucell, Cs_ao, psi_ks_in, nocc, nvirt, /*occ_first=*/false, read_from_aims, aims_nbasis)),
            Cs_ov_mo(cal_Cs_mo(ucell, Cs_ao, psi_ks_in, nocc, nvirt, /*occ_first=*/true, read_from_aims, aims_nbasis)),
            CV_vo(cal_CV(Cs_vo_mo, Vs)),
            CV_ov(cal_CV(Cs_ov_mo, Vs))
        {
            ModuleBase::TITLE("OperatorRIHartree", "OperatorRIHartree");
            this->cal_type = hamilt::calculation_type::lcao_gint;
            this->act_type = 2;
            this->is_first_node = true;
            // check: compare with CVC
            std::vector<T> Amat1 = cal_Amat_full(Cs_vo_mo, Cs_vo_mo, Vs);
            std::vector<T> Amat2 = cal_Amat_full(Cs_vo_mo, Cs_ov_mo, Vs);
            std::vector<T> Amat3 = cal_Amat_full(Cs_ov_mo, Cs_vo_mo, Vs);
            std::vector<T> Amat4 = cal_Amat_full(Cs_ov_mo, Cs_ov_mo, Vs);
            std::vector<T> Amat(Amat1.size());
            for (int i = 0;i < Amat1.size();++i)
            {
                Amat[i] = Amat1[i] + Amat2[i] + Amat3[i] + Amat4[i];
            }
            std::cout << "Amat_full (Hartree term) from RI (Unit Hartree):" << std::endl;
            for (int i = 0;i < npairs;++i)
            {
                for (int j = 0;j < npairs;++j)
                {
                    std::cout << Amat[i * npairs + j] << " ";
                }
                std::cout << std::endl;
            }
        };
        ~OperatorRIHartree() {}
        void act(const psi::Psi<T>& X_in, psi::Psi<T>& X_out, const int nbands) const override
        {
            assert(GlobalV::MY_RANK == 0);  // only serial now
            const int nk = 1;
            const psi::Psi<T>& X = LR_Util::k1_to_bfirst_wrapper(X_in, nk, npairs);
            psi::Psi<T> AX = LR_Util::k1_to_bfirst_wrapper(X_out, nk, npairs);
            for (int ib = 0;ib < nbands;++ib)
            {
                TLRIX<T> CsX_vo = cal_CsX(Cs_vo_mo, &X(ib, 0, 0));
                TLRIX<T> CsX_ov = cal_CsX(Cs_ov_mo, &X(ib, 0, 0));
                // LR_Util::print_CsX(Cs_bX, nvirt, "Cs_bX of state " + std::to_string(ib));
                cal_AX(CV_vo, CsX_vo, &AX(ib, 0, 0), 4.);
                cal_AX(CV_vo, CsX_ov, &AX(ib, 0, 0), 4.);
                cal_AX(CV_ov, CsX_vo, &AX(ib, 0, 0), 4.);
                cal_AX(CV_ov, CsX_ov, &AX(ib, 0, 0), 4.);
            }
        }
    protected:
        const int& naos;
        const int& nocc;
        const int& nvirt;
        const int npairs;
        const TLRI<TR> Cs_ao;
        const TLRI<TR> Vs;
        const psi::Psi<T>& psi_ks;
        const TLRI<T> Cs_ov_mo;
        const TLRI<T> Cs_vo_mo;
        const TLRI<T> CV_ov;
        const TLRI<T> CV_vo;
    };
}
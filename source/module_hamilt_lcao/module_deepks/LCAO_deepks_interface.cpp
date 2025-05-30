#ifdef __DEEPKS
#include "LCAO_deepks_interface.h"

#include "LCAO_deepks_io.h" // mohan add 2024-07-22
#include "module_base/global_variable.h"
#include "module_base/tool_title.h"
#include "module_elecstate/cal_dm.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer.h"
#include "module_hamilt_lcao/module_hcontainer/output_hcontainer.h"
#include "module_parameter/parameter.h"

template <typename TK, typename TR>
LCAO_Deepks_Interface<TK, TR>::LCAO_Deepks_Interface(std::shared_ptr<LCAO_Deepks<TK>> ld_in) : ld(ld_in)
{
}

template <typename TK, typename TR>
void LCAO_Deepks_Interface<TK, TR>::out_deepks_labels(const double& etot,
                                                      const int& nks,
                                                      const int& nat,
                                                      const int& nlocal,
                                                      const ModuleBase::matrix& ekb,
                                                      const std::vector<ModuleBase::Vector3<double>>& kvec_d,
                                                      const UnitCell& ucell,
                                                      const LCAO_Orbitals& orb,
                                                      const Grid_Driver& GridD,
                                                      const Parallel_Orbitals* ParaV,
                                                      const psi::Psi<TK>& psi,
                                                      const elecstate::DensityMatrix<TK, double>* dm,
                                                      hamilt::HamiltLCAO<TK, TR>* p_ham,
                                                      const int rank)
{
    ModuleBase::TITLE("LCAO_Deepks_Interface", "out_deepks_labels");
    ModuleBase::timer::tick("LCAO_Deepks_Interface", "out_deepks_labels");

    // Note: out_deepks_labels does not support equivariant version now!

    // define TH for different types
    using TH = std::conditional_t<std::is_same<TK, double>::value, ModuleBase::matrix, ModuleBase::ComplexMatrix>;

    // These variables are frequently used in the following code
    const int nlmax = orb.Alpha[0].getTotal_nchi();
    const int inlmax = nlmax * nat;
    const int lmaxd = orb.get_lmax_d();
    const int nmaxd = ld->nmaxd;

    const int des_per_atom = ld->des_per_atom;
    const std::vector<int> inl2l = ld->inl2l;
    const ModuleBase::IntArray* inl_index = ld->inl_index;
    const std::vector<hamilt::HContainer<double>*> phialpha = ld->phialpha;

    std::vector<torch::Tensor> pdm = ld->pdm;
    bool init_pdm = ld->init_pdm;
    double E_delta = ld->E_delta;
    double e_delta_band = ld->e_delta_band;
    hamilt::HContainer<double>* dmr = ld->dm_r;

    const int nspin = PARAM.inp.nspin;
    const int nk = nks / nspin;

    // Note : update PDM and all other quantities with the current dm
    // DeePKS PDM and descriptor
    if (PARAM.inp.deepks_out_labels || PARAM.inp.deepks_scf)
    {
        // this part is for integrated test of deepks
        // so it is printed no matter even if deepks_out_labels is not used
        DeePKS_domain::update_dmr(kvec_d, dm->get_DMK_vector(), ucell, orb, *ParaV, GridD, dmr);

        DeePKS_domain::cal_pdm<
            TK>(init_pdm, inlmax, lmaxd, inl2l, inl_index, kvec_d, dmr, phialpha, ucell, orb, GridD, *ParaV, pdm);

        DeePKS_domain::check_pdm(inlmax, inl2l, pdm); // print out the projected dm for NSCF calculaiton

        std::vector<torch::Tensor> descriptor;
        DeePKS_domain::cal_descriptor(nat, inlmax, inl2l, pdm, descriptor,
                                      des_per_atom); // final descriptor
        DeePKS_domain::check_descriptor(inlmax,
                                        des_per_atom,
                                        inl2l,
                                        ucell,
                                        PARAM.globalv.global_out_dir,
                                        descriptor,
                                        rank);

        if (PARAM.inp.deepks_out_labels)
        {
            LCAO_deepks_io::save_npy_d(nat,
                                       des_per_atom,
                                       inlmax,
                                       inl2l,
                                       PARAM.inp.deepks_equiv,
                                       descriptor,
                                       PARAM.globalv.global_out_dir,
                                       rank); // libnpy needed
        }

        if (PARAM.inp.deepks_scf)
        {
            // update E_delta and gedm
            // new gedm is also useful in cal_f_delta, so it should be ld->gedm
            if (PARAM.inp.deepks_equiv)
            {
                DeePKS_domain::cal_edelta_gedm_equiv(nat,
                                                     lmaxd,
                                                     nmaxd,
                                                     inlmax,
                                                     des_per_atom,
                                                     inl2l,
                                                     descriptor,
                                                     ld->gedm,
                                                     E_delta,
                                                     rank);
            }
            else
            {
                DeePKS_domain::cal_edelta_gedm(nat,
                                               inlmax,
                                               des_per_atom,
                                               inl2l,
                                               descriptor,
                                               pdm,
                                               ld->model_deepks,
                                               ld->gedm,
                                               E_delta);
            }
        }
    }

    // Used for deepks_bandgap == 1 and deepks_v_delta > 0
    std::vector<std::vector<TK>>* h_delta = &ld->V_delta;

    // calculating deepks correction and save the results
    if (PARAM.inp.deepks_out_labels)
    {
        // Used for deepks_scf == 1
        std::vector<torch::Tensor> gevdm;
        if (PARAM.inp.deepks_scf)
        {
            DeePKS_domain::cal_gevdm(nat, inlmax, inl2l, pdm, gevdm);
        }

        // Energy Part
        const std::string file_etot = PARAM.globalv.global_out_dir + "deepks_etot.npy";
        const std::string file_ebase = PARAM.globalv.global_out_dir + "deepks_ebase.npy";

        LCAO_deepks_io::save_npy_e(etot, file_etot, rank);

        if (PARAM.inp.deepks_scf)
        {
            /// ebase :no deepks E_delta including
            LCAO_deepks_io::save_npy_e(etot - E_delta, file_ebase, rank);
        }
        else // deepks_scf = 0; base calculation
        {
            /// no scf, e_tot=e_base
            LCAO_deepks_io::save_npy_e(etot, file_ebase, rank);
        }

        // Force Part
        if (PARAM.inp.cal_force)
        {
            if (PARAM.inp.deepks_scf
                && !PARAM.inp.deepks_equiv) // training with force label not supported by equivariant version now
            {
                torch::Tensor gdmx;
                DeePKS_domain::cal_gdmx<
                    TK>(lmaxd, inlmax, nks, kvec_d, phialpha, inl_index, dmr, ucell, orb, *ParaV, GridD, gdmx);

                torch::Tensor gvx;
                DeePKS_domain::cal_gvx(ucell.nat, inlmax, des_per_atom, inl2l, gevdm, gdmx, gvx, rank);
                const std::string file_gradvx = PARAM.globalv.global_out_dir + "deepks_gradvx.npy";
                LCAO_deepks_io::save_tensor2npy<double>(file_gradvx, gvx, rank);

                if (PARAM.inp.deepks_out_unittest)
                {
                    DeePKS_domain::check_gdmx(gdmx);
                    DeePKS_domain::check_gvx(gvx, rank);
                }
            }
        }

        // Stress Part
        if (PARAM.inp.cal_stress)
        {
            if (PARAM.inp.deepks_scf
                && !PARAM.inp.deepks_equiv) // training with stress label not supported by equivariant version now
            {
                torch::Tensor gdmepsl;
                DeePKS_domain::cal_gdmepsl<
                    TK>(lmaxd, inlmax, nks, kvec_d, phialpha, inl_index, dmr, ucell, orb, *ParaV, GridD, gdmepsl);

                torch::Tensor gvepsl;
                DeePKS_domain::cal_gvepsl(ucell.nat, inlmax, des_per_atom, inl2l, gevdm, gdmepsl, gvepsl, rank);
                const std::string file_gvepsl = PARAM.globalv.global_out_dir + "deepks_gvepsl.npy";
                LCAO_deepks_io::save_tensor2npy<double>(file_gvepsl, gvepsl, rank);

                if (PARAM.inp.deepks_out_unittest)
                {
                    DeePKS_domain::check_gdmepsl(gdmepsl);
                    DeePKS_domain::check_gvepsl(gvepsl, rank);
                }
            }
        }

        // Bandgap Part
        if (PARAM.inp.deepks_bandgap)
        {
            const int nocc = (PARAM.inp.nelec + 1) / 2;
            ModuleBase::matrix o_tot(nks, 1);
            for (int iks = 0; iks < nks; ++iks)
            {
                // record band gap for each k point (including spin)
                o_tot(iks, 0) = ekb(iks, nocc) - ekb(iks, nocc - 1);
            }

            const std::string file_otot = PARAM.globalv.global_out_dir + "deepks_otot.npy";
            LCAO_deepks_io::save_matrix2npy(file_otot, o_tot, rank); // Unit: Ry

            if (PARAM.inp.deepks_scf)
            {
                ModuleBase::matrix wg_hl;
                std::vector<TH> dm_bandgap;

                // Calculate O_delta
                wg_hl.create(nks, PARAM.inp.nbands);
                dm_bandgap.resize(nks);
                wg_hl.zero_out();
                for (int iks = 0; iks < nks; ++iks)
                {
                    wg_hl(iks, nocc - 1) = -1.0;
                    wg_hl(iks, nocc) = 1.0;
                }
                elecstate::cal_dm(ParaV, wg_hl, psi, dm_bandgap);

                ModuleBase::matrix o_delta(nks, 1);

                // calculate and save orbital_precalc: [nks,NAt,NDscrpt]
                torch::Tensor orbital_precalc;
                DeePKS_domain::cal_orbital_precalc<TK, TH>(dm_bandgap,
                                                           lmaxd,
                                                           inlmax,
                                                           nat,
                                                           nks,
                                                           inl2l,
                                                           kvec_d,
                                                           phialpha,
                                                           gevdm,
                                                           inl_index,
                                                           ucell,
                                                           orb,
                                                           *ParaV,
                                                           GridD,
                                                           orbital_precalc);
                DeePKS_domain::cal_o_delta<TK, TH>(dm_bandgap, *h_delta, o_delta, *ParaV, nks, nspin);

                // save obase and orbital_precalc
                const std::string file_orbpre = PARAM.globalv.global_out_dir + "deepks_orbpre.npy";
                LCAO_deepks_io::save_tensor2npy<double>(file_orbpre, orbital_precalc, rank);

                const std::string file_obase = PARAM.globalv.global_out_dir + "deepks_obase.npy";
                LCAO_deepks_io::save_matrix2npy(file_obase, o_tot - o_delta, rank); // Unit: Ry
            }                                                                       // end deepks_scf == 1
            else                                                                    // deepks_scf == 0
            {
                const std::string file_obase = PARAM.globalv.global_out_dir + "deepks_obase.npy";
                LCAO_deepks_io::save_matrix2npy(file_obase, o_tot, rank); // no scf, o_tot=o_base
            }                                                             // end deepks_scf == 0
        }                                                                 // end bandgap label

        // H(R) matrix part, for HR, base will not be calculated since they are HContainer objects
        if (PARAM.inp.deepks_v_delta < 0)
        {
            // set the output
            const double sparse_threshold = 1e-10;
            const int precision = 8;
            const std::string file_hrtot = PARAM.globalv.global_out_dir + "deepks_hrtot.csr";
            hamilt::HContainer<TR>* hR_tot = (p_ham->getHR());

            if (rank == 0)
            {
                std::ofstream ofs_hr(file_hrtot, std::ios::out);
                ofs_hr << "Matrix Dimension of H(R): " << hR_tot->get_nbasis() << std::endl;
                ofs_hr << "Matrix number of H(R): " << hR_tot->size_R_loop() << std::endl;
                hamilt::Output_HContainer<TR> out_hr(hR_tot, ofs_hr, sparse_threshold, precision);
                out_hr.write();
                ofs_hr.close();
            }

            if (PARAM.inp.deepks_scf)
            {
                const std::string file_vdeltar = PARAM.globalv.global_out_dir + "deepks_hrdelta.csr";
                hamilt::HContainer<TR>* h_deltaR = p_ham->get_V_delta_R();

                if (rank == 0)
                {
                    std::ofstream ofs_hr(file_vdeltar, std::ios::out);
                    ofs_hr << "Matrix Dimension of H_delta(R): " << h_deltaR->get_nbasis() << std::endl;
                    ofs_hr << "Matrix number of H_delta(R): " << h_deltaR->size_R_loop() << std::endl;
                    hamilt::Output_HContainer<TR> out_hr(h_deltaR, ofs_hr, sparse_threshold, precision);
                    out_hr.write();
                    ofs_hr.close();
                }

                torch::Tensor phialpha_r_out;
                torch::Tensor R_query;
                DeePKS_domain::prepare_phialpha_r(nlocal,
                                                  lmaxd,
                                                  inlmax,
                                                  nat,
                                                  phialpha,
                                                  ucell,
                                                  orb,
                                                  *ParaV,
                                                  GridD,
                                                  phialpha_r_out,
                                                  R_query);
                const std::string file_phialpha_r = PARAM.globalv.global_out_dir + "deepks_phialpha_r.npy";
                const std::string file_R_query = PARAM.globalv.global_out_dir + "deepks_R_query.npy";
                LCAO_deepks_io::save_tensor2npy<double>(file_phialpha_r, phialpha_r_out, rank);
                LCAO_deepks_io::save_tensor2npy<int>(file_R_query, R_query, rank);

                torch::Tensor gevdm_out;
                DeePKS_domain::prepare_gevdm(nat, lmaxd, inlmax, orb, gevdm, gevdm_out);
                const std::string file_gevdm = PARAM.globalv.global_out_dir + "deepks_gevdm.npy";
                LCAO_deepks_io::save_tensor2npy<double>(file_gevdm, gevdm_out, rank);
            }
        }

        // H(k) matrix part
        if (PARAM.inp.deepks_v_delta > 0)
        {
            std::vector<TH> h_tot(nks);
            std::vector<std::vector<TK>> h_mat(nks, std::vector<TK>(ParaV->nloc));
            for (int ik = 0; ik < nks; ik++)
            {
                h_tot[ik].create(nlocal, nlocal);
                p_ham->updateHk(ik);
                const TK* hk_ptr = p_ham->getHk();
                for (int i = 0; i < ParaV->nloc; i++)
                {
                    h_mat[ik][i] = hk_ptr[i];
                }
            }

            DeePKS_domain::collect_h_mat<TK, TH>(*ParaV, h_mat, h_tot, nlocal, nks);

            const std::string file_htot = PARAM.globalv.global_out_dir + "deepks_htot.npy";
            LCAO_deepks_io::save_npy_h<TK, TH>(h_tot, file_htot, nlocal, nks, rank);

            if (PARAM.inp.deepks_scf)
            {
                std::vector<TH> v_delta(nks);
                std::vector<TH> h_base(nks);
                for (int ik = 0; ik < nks; ik++)
                {
                    v_delta[ik].create(nlocal, nlocal);
                    h_base[ik].create(nlocal, nlocal);
                }
                DeePKS_domain::collect_h_mat<TK, TH>(*ParaV, *h_delta, v_delta, nlocal, nks);

                // save v_delta and h_base
                const std::string file_hbase = PARAM.globalv.global_out_dir + "deepks_hbase.npy";
                for (int ik = 0; ik < nks; ik++)
                {
                    h_base[ik] = h_tot[ik] - v_delta[ik];
                }
                LCAO_deepks_io::save_npy_h<TK, TH>(h_base, file_hbase, nlocal, nks, rank);

                const std::string file_vdelta = PARAM.globalv.global_out_dir + "deepks_vdelta.npy";
                LCAO_deepks_io::save_npy_h<TK, TH>(v_delta, file_vdelta, nlocal, nks, rank);

                if (PARAM.inp.deepks_v_delta == 1) // v_delta_precalc storage method 1
                {
                    torch::Tensor v_delta_precalc;
                    DeePKS_domain::cal_v_delta_precalc<TK>(nlocal,
                                                           lmaxd,
                                                           inlmax,
                                                           nat,
                                                           nks,
                                                           inl2l,
                                                           kvec_d,
                                                           phialpha,
                                                           gevdm,
                                                           inl_index,
                                                           ucell,
                                                           orb,
                                                           *ParaV,
                                                           GridD,
                                                           v_delta_precalc);

                    const std::string file_vdpre = PARAM.globalv.global_out_dir + "deepks_vdpre.npy";
                    LCAO_deepks_io::save_tensor2npy<TK>(file_vdpre, v_delta_precalc, rank);
                }
                else if (PARAM.inp.deepks_v_delta == 2) // v_delta_precalc storage method 2
                {
                    torch::Tensor phialpha_out;
                    DeePKS_domain::prepare_phialpha<
                        TK>(nlocal, lmaxd, inlmax, nat, nks, kvec_d, phialpha, ucell, orb, *ParaV, GridD, phialpha_out);
                    const std::string file_phialpha = PARAM.globalv.global_out_dir + "deepks_phialpha.npy";
                    LCAO_deepks_io::save_tensor2npy<TK>(file_phialpha, phialpha_out, rank);

                    torch::Tensor gevdm_out;
                    DeePKS_domain::prepare_gevdm(nat, lmaxd, inlmax, orb, gevdm, gevdm_out);
                    const std::string file_gevdm = PARAM.globalv.global_out_dir + "deepks_gevdm.npy";
                    LCAO_deepks_io::save_tensor2npy<double>(file_gevdm, gevdm_out, rank);
                }
            }
            else // deepks_scf == 0
            {
                const std::string file_hbase = PARAM.globalv.global_out_dir + "deepks_hbase.npy";
                LCAO_deepks_io::save_npy_h<TK, TH>(h_tot, file_hbase, nlocal, nks, rank);
            }
        } // end v_delta label

    } // end deepks_out_labels

    /// print out deepks information to the screen
    if (PARAM.inp.deepks_scf)
    {
        DeePKS_domain::cal_e_delta_band(dm->get_DMK_vector(), *h_delta, nks, nspin, ParaV, e_delta_band);
        std::cout << "E_delta_band = " << std::setprecision(8) << e_delta_band << " Ry"
                  << " = " << std::setprecision(8) << e_delta_band * ModuleBase::Ry_to_eV << " eV" << std::endl;
        std::cout << "E_delta_NN = " << std::setprecision(8) << E_delta << " Ry"
                  << " = " << std::setprecision(8) << E_delta * ModuleBase::Ry_to_eV << " eV" << std::endl;
        if (PARAM.inp.deepks_out_unittest)
        {
            LCAO_deepks_io::print_dm(nks, PARAM.globalv.nlocal, ParaV->nrow, dm->get_DMK_vector());

            DeePKS_domain::check_gedm(inlmax, inl2l, ld->gedm);

            std::ofstream ofs("E_delta_bands.dat");
            ofs << std::setprecision(10) << e_delta_band;

            std::ofstream ofs1("E_delta.dat");
            ofs1 << std::setprecision(10) << E_delta;
        }
    }
    ModuleBase::timer::tick("LCAO_Deepks_Interface", "out_deepks_labels");
}

template class LCAO_Deepks_Interface<double, double>;
template class LCAO_Deepks_Interface<std::complex<double>, double>;
template class LCAO_Deepks_Interface<std::complex<double>, std::complex<double>>;

#endif

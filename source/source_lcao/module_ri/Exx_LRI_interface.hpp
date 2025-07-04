#ifndef EXX_LRI_INTERFACE_HPP
#define EXX_LRI_INTERFACE_HPP
#include "module_parameter/parameter.h"

#include "Exx_LRI_interface.h"
#include "source_lcao/module_ri/exx_abfs-jle.h"
#include "source_lcao/module_ri/exx_opt_orb.h"
#include "source_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "source_lcao/hamilt_lcaodft/operator_lcao/op_exx_lcao.h"
#include "source_base/parallel_common.h"
#include "source_base/formatter.h"

#include "source_io/csr_reader.h"
#include "source_io/write_HS_sparse.h"
#include "source_estate/elecstate_lcao.h"

#include <sys/time.h>
#include <stdexcept>
#include <string>

/*
template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::write_Hexxs_cereal(const std::string& file_name) const
{
    ModuleBase::TITLE("Exx_LRI_Interface", "write_Hexxs_cereal");
    ModuleBase::timer::tick("Exx_LRI_Interface", "write_Hexxs_cereal");
    std::ofstream ofs(file_name + "_" + std::to_string(GlobalV::MY_RANK), std::ofstream::binary);
    cereal::BinaryOutputArchive oar(ofs);
    oar(this->exx_ptr->Hexxs);
    ModuleBase::timer::tick("Exx_LRI_Interface", "write_Hexxs_cereal");
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::read_Hexxs_cereal(const std::string& file_name)
{
    ModuleBase::TITLE("Exx_LRI_Interface", "read_Hexxs_cereal");
    ModuleBase::timer::tick("Exx_LRI_Interface", "read_Hexxs_cereal");
    const std::string file_name_rank = file_name + "_" + std::to_string(GlobalV::MY_RANK);
    std::ifstream ifs(file_name_rank, std::ofstream::binary);
    if(!ifs.is_open())
        { ModuleBase::WARNING_QUIT("Exx_LRI_Interface", file_name_rank+" not found."); }
    cereal::BinaryInputArchive iar(ifs);
    iar(this->exx_ptr->Hexxs);
    ModuleBase::timer::tick("Exx_LRI_Interface", "read_Hexxs_cereal");
}
*/

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::init(const MPI_Comm &mpi_comm,
                                       const UnitCell &ucell,
                                       const K_Vectors &kv,
                                       const LCAO_Orbitals& orb)
{
    ModuleBase::TITLE("Exx_LRI_Interface","init");
    this->exx_ptr->init(mpi_comm, ucell, kv, orb);
    this->flag_finish.init = true;
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::cal_exx_ions(const UnitCell& ucell, const bool write_cv)
{
    ModuleBase::TITLE("Exx_LRI_Interface","cal_exx_ions");
    if(!this->flag_finish.init)
        { throw std::runtime_error("Exx init unfinished when "+std::string(__FILE__)+" line "+std::to_string(__LINE__)); }

    this->exx_ptr->cal_exx_ions(ucell, write_cv);

    this->flag_finish.ions = true;
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::cal_exx_elec(const std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>>& Ds,
                                               const UnitCell& ucell,
                                               const Parallel_Orbitals& pv,
                                               const ModuleSymmetry::Symmetry_rotation* p_symrot)
{
    ModuleBase::TITLE("Exx_LRI_Interface","cal_exx_elec");
    if(!this->flag_finish.init || !this->flag_finish.ions)
    { 
        throw std::runtime_error("Exx init unfinished when "
        +std::string(__FILE__)+" line "+std::to_string(__LINE__)); 
    }

    this->exx_ptr->cal_exx_elec(Ds, ucell, pv, p_symrot);

    this->flag_finish.elec = true;
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::cal_exx_force(const int& nat)
{
    ModuleBase::TITLE("Exx_LRI_Interface","cal_exx_force");
    if(!this->flag_finish.init || !this->flag_finish.ions)
    { 
        throw std::runtime_error("Exx init unfinished when "+std::string(__FILE__)+" line "+std::to_string(__LINE__)); 
    }
    if(!this->flag_finish.elec)
    { 
        throw std::runtime_error("Exx Hamiltonian unfinished when "+std::string(__FILE__)
        +" line "+std::to_string(__LINE__)); 
    }

    this->exx_ptr->cal_exx_force(nat);

    this->flag_finish.force = true;
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::cal_exx_stress(const double& omega, const double& lat0)
{
    ModuleBase::TITLE("Exx_LRI_Interface","cal_exx_stress");
    if(!this->flag_finish.init || !this->flag_finish.ions)
    { 
        throw std::runtime_error("Exx init unfinished when "
                +std::string(__FILE__)+" line "+std::to_string(__LINE__)); 
    }
    if(!this->flag_finish.elec)
    { 
        throw std::runtime_error("Exx Hamiltonian unfinished when "
                +std::string(__FILE__)+" line "+std::to_string(__LINE__)); 
    }

    this->exx_ptr->cal_exx_stress(omega, lat0);

    this->flag_finish.stress = true;
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::exx_before_all_runners(
    const K_Vectors& kv, 
    const UnitCell& ucell,
    const Parallel_2D& pv)
{
    ModuleBase::TITLE("Exx_LRI_Interface","exx_before_all_runners");
    // initialize the rotation matrix in AO representation
    this->exx_spacegroup_symmetry = (PARAM.inp.nspin < 4 && ModuleSymmetry::Symmetry::symm_flag == 1);
    if (this->exx_spacegroup_symmetry)
    {
        const std::array<int, 3>& period = RI_Util::get_Born_vonKarmen_period(kv);
        this->symrot_.find_irreducible_sector(
            ucell.symm, ucell.atoms, ucell.st,
            RI_Util::get_Born_von_Karmen_cells(period), period, ucell.lat);
        // this->symrot_.set_Cs_rotation(this->exx_ptr->get_abfs_nchis());
        this->symrot_.cal_Ms(kv, ucell, pv);
    }
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::exx_beforescf(const int istep,
                                                const K_Vectors& kv,
                                                const Charge_Mixing& chgmix,
                                                const UnitCell& ucell,
                                                const LCAO_Orbitals& orb)
{
    ModuleBase::TITLE("Exx_LRI_Interface","exx_beforescf");
#ifdef __MPI
    if (GlobalC::exx_info.info_global.cal_exx)
    {
        if ((GlobalC::restart.info_load.load_H_finish && !GlobalC::restart.info_load.restart_exx)
            || (istep > 0)
            || (PARAM.inp.init_wfc == "file"))
        {
            XC_Functional::set_xc_type(ucell.atoms[0].ncpp.xc_func);
        }
        else
        {
            if (ucell.atoms[0].ncpp.xc_func == "HF" || ucell.atoms[0].ncpp.xc_func == "PBE0" || ucell.atoms[0].ncpp.xc_func == "HSE")
            {
                XC_Functional::set_xc_type("pbe");
            }
            else if (ucell.atoms[0].ncpp.xc_func == "SCAN0")
            {
                XC_Functional::set_xc_type("scan");
            }
            // added by jghan, 2024-07-07
            else if ( ucell.atoms[0].ncpp.xc_func == "MULLER" || ucell.atoms[0].ncpp.xc_func == "POWER"
                || ucell.atoms[0].ncpp.xc_func == "WP22" || ucell.atoms[0].ncpp.xc_func == "CWP22" )
            {
                XC_Functional::set_xc_type("pbe");
            }
        }

        this->cal_exx_ions(ucell,PARAM.inp.out_ri_cv);
    }

    if (Exx_Abfs::Jle::generate_matrix)
    {
        //program should be stopped after this judgement
        Exx_Opt_Orb exx_opt_orb;
        exx_opt_orb.generate_matrix(kv, ucell,orb);
        ModuleBase::timer::tick("ESolver_KS_LCAO", "beforescf");
        return;
    }

    // set initial parameter for mix_DMk_2D
    if(GlobalC::exx_info.info_global.cal_exx)
    {
        if (this->exx_spacegroup_symmetry)
            {this->mix_DMk_2D.set_nks(kv.get_nkstot_full() * (PARAM.inp.nspin == 2 ? 2 : 1), PARAM.globalv.gamma_only_local);}
        else
            {this->mix_DMk_2D.set_nks(kv.get_nks(), PARAM.globalv.gamma_only_local);}

        if(GlobalC::exx_info.info_global.separate_loop)
            { this->mix_DMk_2D.set_mixing(nullptr); }
        else
            { this->mix_DMk_2D.set_mixing(chgmix.get_mixing()); }
        // for exx two_level scf
        this->two_level_step = 0;
    }
#endif // __MPI
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::exx_eachiterinit(const int istep,
                                                   const UnitCell& ucell,
                                                   const elecstate::DensityMatrix<T, double>& dm,
                                                   const K_Vectors& kv,
                                                   const int& iter)
{
    ModuleBase::TITLE("Exx_LRI_Interface","exx_eachiterinit");
    if (GlobalC::exx_info.info_global.cal_exx)
    {
        if (!GlobalC::exx_info.info_global.separate_loop 
            && (this->two_level_step 
                || istep > 0 
                || PARAM.inp.init_wfc == "file") // non separate loop case
            || (GlobalC::exx_info.info_global.separate_loop 
                && PARAM.inp.init_wfc == "file" 
                && this->two_level_step == 0 
                && iter == 1)
           )  // the first iter in separate loop case
        {
            const bool flag_restart = (iter == 1) ? true : false;

            auto cal = [this, &ucell,&kv, &flag_restart](const elecstate::DensityMatrix<T, double>& dm_in)
            {
                if (this->exx_spacegroup_symmetry)
                    { this->mix_DMk_2D.mix(symrot_.restore_dm(kv,dm_in.get_DMK_vector(), *dm_in.get_paraV_pointer()), flag_restart); }
                else
                    { this->mix_DMk_2D.mix(dm_in.get_DMK_vector(), flag_restart); }
                const std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>>
                    Ds = PARAM.globalv.gamma_only_local
                        ? RI_2D_Comm::split_m2D_ktoR<Tdata>(
                            ucell,
                            *this->exx_ptr->p_kv,
                            this->mix_DMk_2D.get_DMk_gamma_out(),
                            *dm_in.get_paraV_pointer(),
                            PARAM.inp.nspin)
                        : RI_2D_Comm::split_m2D_ktoR<Tdata>(
                            ucell,
                            *this->exx_ptr->p_kv,
                            this->mix_DMk_2D.get_DMk_k_out(),
                            *dm_in.get_paraV_pointer(),
                            PARAM.inp.nspin,
                            this->exx_spacegroup_symmetry);

                if (this->exx_spacegroup_symmetry && GlobalC::exx_info.info_ri.exx_symmetry_realspace)
                    { this->cal_exx_elec(Ds, ucell,*dm_in.get_paraV_pointer(), &this->symrot_); }
                else
                    { this->cal_exx_elec(Ds, ucell,*dm_in.get_paraV_pointer()); }
            };

            if(istep > 0 && flag_restart)
                { cal(*dm_last_step); }
            else
                { cal(dm); }
        }
    }
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::exx_hamilt2rho(elecstate::ElecState& elec, const Parallel_Orbitals& pv, const  int iter)
{
    ModuleBase::TITLE("Exx_LRI_Interface","exx_hamilt2density");
    // Peize Lin add 2020.04.04
    if (XC_Functional::get_func_type() == 4 || XC_Functional::get_func_type() == 5)
    {
        // add exx
        // Peize Lin add 2016-12-03
        if (GlobalC::restart.info_load.load_H_finish && !GlobalC::restart.info_load.restart_exx
            && this->two_level_step == 0 && iter == 1)
        {
            if (GlobalV::MY_RANK == 0)
            {
                try
                    { GlobalC::restart.load_disk("Eexx", 0, 1, &this->exx_ptr->Eexx); }
                catch (const std::exception& e)
                    { std::cout << "WARNING: Cannot read Eexx from disk, the energy of the 1st loop will be wrong, sbut it does not influence the subsequent loops." << std::endl; }
            }
            Parallel_Common::bcast_double(this->exx_ptr->Eexx);
            this->exx_ptr->Eexx /= GlobalC::exx_info.info_global.hybrid_alpha;
        }
        elec.set_exx(this->get_Eexx());
    }
    else
    {
        elec.f_en.exx = 0.;
    }
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::exx_iter_finish(const K_Vectors& kv,
                                                  const UnitCell& ucell,
                                                  hamilt::Hamilt<T>& hamilt,
                                                  elecstate::ElecState& elec,
                                                  Charge_Mixing& chgmix,
                                                  const double& scf_ene_thr,
                                                  int& iter,
                                                  const int istep,
                                                  bool& conv_esolver)
{
    ModuleBase::TITLE("Exx_LRI_Interface","exx_iter_finish");
    if (GlobalC::restart.info_save.save_H && (this->two_level_step > 0 || istep > 0)
        && (!GlobalC::exx_info.info_global.separate_loop || iter == 1)) // to avoid saving the same value repeatedly
    {
        ////////// for Add_Hexx_Type::k
        /*
        hamilt::HS_Matrix_K<TK> Hexxk_save(&this->pv, 1);
        for (int ik = 0; ik < this->kv.get_nks(); ++ik) {
            Hexxk_save.set_zero_hk();

            hamilt::OperatorEXX<hamilt::OperatorLCAO<TK, TR>> opexx_save(&Hexxk_save,
                                                                         nullptr,
                                                                         this->kv);

            opexx_save.contributeHk(ik);

            GlobalC::restart.save_disk("Hexx",
                                       ik,
                                       this->pv.get_local_size(),
                                       Hexxk_save.get_hk());
        }*/
        ////////// for Add_Hexx_Type:R
        const std::string& restart_HR_path = GlobalC::restart.folder + "HexxR" + std::to_string(GlobalV::MY_RANK);
        ModuleIO::write_Hexxs_csr(restart_HR_path, ucell, this->get_Hexxs());

        if (GlobalV::MY_RANK == 0)
        {
            GlobalC::restart.save_disk("Eexx", 0, 1, &elec.f_en.exx);
        }
    }

    if (GlobalC::exx_info.info_global.cal_exx && conv_esolver)
    {
        // Kerker mixing does not work for the density matrix.
        // In the separate loop case, it can still work in the subsequent inner loops where Hexx(DM) is fixed.
        // In the non-separate loop case where Hexx(DM) is updated in every iteration of the 2nd loop, it should be
        // closed.
        if (!GlobalC::exx_info.info_global.separate_loop)
        {
            chgmix.close_kerker_gg0();
        }
        this->dm_last_step = dynamic_cast<const elecstate::ElecStateLCAO<T>*>(&elec)->get_DM();
        conv_esolver = this->exx_after_converge(
            ucell,
            hamilt,
            *dynamic_cast<const elecstate::ElecStateLCAO<T>*>(&elec)->get_DM(),
            kv,
            PARAM.inp.nspin,
            iter,
            istep,
            elec.f_en.etot,
            scf_ene_thr);
    }
    //else if ( PARAM.inp.rdmft && two_level_step ) { conv_esolver = true; }    // for RDMFT in the future to quit after the first iter of the exx-loop
}

template<typename T, typename Tdata>
bool Exx_LRI_Interface<T, Tdata>::exx_after_converge(
    const UnitCell& ucell,
    hamilt::Hamilt<T>& hamilt,
    const elecstate::DensityMatrix<T, double>& dm,
    const K_Vectors& kv,
    const int& nspin,
    int& iter,
    const int& istep,
    const double& etot,
    const double& scf_ene_thr)
{   // only called if (GlobalC::exx_info.info_global.cal_exx)
    ModuleBase::TITLE("Exx_LRI_Interface","exx_after_converge");
    auto restart_reset = [this]()
    { // avoid calling restart related procedure in the subsequent ion steps
        GlobalC::restart.info_load.restart_exx = true;
        this->exx_ptr->Eexx = 0;
    };

    // no separate_loop case
    if (!GlobalC::exx_info.info_global.separate_loop)
    {
        GlobalC::exx_info.info_global.hybrid_step = 1;

        // in no_separate_loop case, scf loop only did twice
        // in first scf loop, exx updated once in beginning,
        // in second scf loop, exx updated every iter

        if (this->two_level_step || istep > 0)
        {
            restart_reset();
            return true;
        }
        else
        {
            // update exx and redo scf
            XC_Functional::set_xc_type(ucell.atoms[0].ncpp.xc_func);
            iter = 0;
            std::cout << " Entering 2nd SCF, where EXX is updated" << std::endl;
            this->two_level_step++;
            return false;
        }
    }
    else
    { // has separate_loop case
        const double ediff = std::abs(etot - etot_last_outer_loop) * ModuleBase::Ry_to_eV;
        if (two_level_step)
            { std::cout << FmtCore::format(" deltaE (eV) from outer loop: %.8e \n", ediff); }
        // exx converged or get max exx steps
        if (this->two_level_step == GlobalC::exx_info.info_global.hybrid_step
            || (iter == 1 && this->two_level_step != 0) // density convergence of outer loop
            || (ediff < scf_ene_thr && this->two_level_step != 0))   //energy convergence of outer loop
        {
            restart_reset();
            return true;
        }
        else
        {
            this->etot_last_outer_loop = etot;
            // update exx and redo scf
            if (this->two_level_step == 0)
                { XC_Functional::set_xc_type(ucell.atoms[0].ncpp.xc_func); }

            std::cout << " Updating EXX " << std::flush;
            timeval t_start;       gettimeofday(&t_start, nullptr);

            // if init_wfc == "file", DM is calculated in the 1st iter of the 1st two-level step, so we mix it here
            const bool flag_restart = (this->two_level_step == 0 && PARAM.inp.init_wfc != "file") ? true : false;

            if (this->exx_spacegroup_symmetry)
                {this->mix_DMk_2D.mix(symrot_.restore_dm(kv, dm.get_DMK_vector(), *dm.get_paraV_pointer()), flag_restart);}
            else
                {this->mix_DMk_2D.mix(dm.get_DMK_vector(), flag_restart);}

            // GlobalC::exx_lcao.cal_exx_elec(p_esolver->LOC, p_esolver->LOWF.wfc_k_grid);
            const std::vector<std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>>>
                Ds = std::is_same<T, double>::value //gamma_only_local
                ? RI_2D_Comm::split_m2D_ktoR<Tdata>(ucell,*this->exx_ptr->p_kv, this->mix_DMk_2D.get_DMk_gamma_out(), *dm.get_paraV_pointer(), nspin)
                : RI_2D_Comm::split_m2D_ktoR<Tdata>(ucell,*this->exx_ptr->p_kv, this->mix_DMk_2D.get_DMk_k_out(), *dm.get_paraV_pointer(), nspin, this->exx_spacegroup_symmetry);

            if (this->exx_spacegroup_symmetry && GlobalC::exx_info.info_ri.exx_symmetry_realspace)
                { this->cal_exx_elec(Ds, ucell, *dm.get_paraV_pointer(), &this->symrot_); }
            else
                { this->cal_exx_elec(Ds, ucell, *dm.get_paraV_pointer()); }    // restore DM but not Hexx
            iter = 0;
            this->two_level_step++;

            timeval t_end;       gettimeofday(&t_end, nullptr);
            std::cout << "and rerun SCF\t"
                << std::setprecision(3) << std::setiosflags(std::ios::scientific)
                << (double)(t_end.tv_sec-t_start.tv_sec) + (double)(t_end.tv_usec-t_start.tv_usec)/1000000.0
                << std::defaultfloat << " (s)" << std::endl;
            return false;
        }
    }   // if(GlobalC::exx_info.info_global.separate_loop)
    restart_reset();
    return true;
}

#endif

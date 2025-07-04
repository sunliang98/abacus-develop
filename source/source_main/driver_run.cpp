#include "source_main/driver.h"
#include "source_cell/check_atomic_stru.h"
#include "source_cell/module_neighbor/sltk_atom_arrange.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "module_parameter/parameter.h"
#include "source_io/para_json.h"
#include "source_io/print_info.h"
#include "source_io/winput.h"
#include "source_md/run_md.h"

/**
 * @brief This is the driver function which defines the workflow of ABACUS
 * calculations. It relies on the class Esolver, which is a class that organizes
 * workflows of single point calculations.
 *
 * For calculations involving change of configuration (lattice parameter & ionic
 * motion), this driver calls Esolver::Run and the configuration-changing
 * subroutine in a alternating manner.
 *
 * Information is passed between the two subroutines by class UnitCell
 *
 * Esolver::Run takes in a configuration and provides force and stress,
 * the configuration-changing subroutine takes force and stress and updates the
 * configuration
 */
void Driver::driver_run()
{
    ModuleBase::TITLE("Driver", "driver_run");

    //! 1: setup cell and atom information
    // this warning should not be here, mohan 2024-05-22
#ifndef __LCAO
    if (PARAM.inp.basis_type == "lcao_in_pw" || PARAM.inp.basis_type == "lcao") {
        ModuleBase::timer::tick("Driver","driver_run");
        ModuleBase::WARNING_QUIT("driver",
                                 "to use LCAO basis, compile with __LCAO");
    }
#endif

    // the life of ucell should begin here, mohan 2024-05-12
    UnitCell ucell;
    ucell.setup(PARAM.inp.latname,
                PARAM.inp.ntype,
                PARAM.inp.lmaxmax,
                PARAM.inp.init_vel,
                PARAM.inp.fixed_axes);

    ucell.setup_cell(PARAM.globalv.global_in_stru, GlobalV::ofs_running);
    unitcell::check_atomic_stru(ucell, PARAM.inp.min_dist_coef);

    //! 2: initialize the ESolver (depends on a set-up ucell after `setup_cell`)
    ModuleESolver::ESolver* p_esolver = ModuleESolver::init_esolver(PARAM.inp, ucell);

    //! 3: initialize Esolver and fill json-structure
    p_esolver->before_all_runners(ucell, PARAM.inp);

    // this Json part should be moved to before_all_runners, mohan 2024-05-12
#ifdef __RAPIDJSON
    Json::gen_stru_wrapper(&ucell);
#endif

    const std::string cal = PARAM.inp.calculation;

    //! 4: different types of calculations
    if (cal == "md")
    {
        Run_MD::md_line(ucell, p_esolver, PARAM);
    }
    else if (cal == "scf" || cal == "relax" || cal == "cell-relax" || cal == "nscf")
    {
        Relax_Driver rl_driver;
        rl_driver.relax_driver(p_esolver, ucell);
    }
    else if (cal == "get_s")
    {
        p_esolver->runner(ucell, 0);
    }
    else if (cal == "get_pchg" || cal == "get_wf" || cal == "gen_bessel" || 
             cal == "test_memory" || cal == "test_neighbour")
    {
        //! supported "other" functions:
        //! get_pchg(LCAO),
        //! test_memory(PW,LCAO),
        //! test_neighbour(LCAO),
        //! gen_bessel(PW), et al.
        const int istep = 0;
        p_esolver->others(ucell, istep);
    }
    else
    {
        ModuleBase::WARNING_QUIT("Driver::driver_run","cannot recognize the 'calculation' command");
    }

    //! 5: clean up esolver
    p_esolver->after_all_runners(ucell);

    ModuleESolver::clean_esolver(p_esolver);

    //! 6: output the json file
    Json::create_Json(&ucell, PARAM);

    return;
}

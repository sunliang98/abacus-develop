#include "relax_sync.h"


#include "source_base/matrix3.h"
#include "source_base/parallel_common.h"
#include "source_base/tool_title.h"
#include "source_cell/update_cell.h"
#include "source_cell/print_cell.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "module_parameter/parameter.h"
#include "source_relax/ions_move_basic.h"

#include <cmath>

void Relax::init_relax(const int nat_in)
{
    ModuleBase::TITLE("Relax", "init_relax");

    // set some initial conditions / constants
    nat = nat_in;
    istep = 0;
    cg_step = 0;
    ltrial = false;
    brent_done = false;
    step_size = 1.0;
    srp_srp = 100000;
    etot = 0;
    etot_p = 0;

    force_thr_eva = PARAM.inp.force_thr * ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A; // convert to eV/A
    fac_force = PARAM.inp.relax_scale_force * 0.1;
    fac_stress = fac_force / nat;

    // allocate some data structures

    // gradients in ion and cell; current and previous
    grad_ion.create(nat, 3);
    grad_cell.create(3, 3);

    grad_ion_p.create(nat, 3);
    grad_cell_p.create(3, 3);

    // search directions for ion and cell; current and previous
    search_dr_ion.create(nat, 3);
    search_dr_cell.create(3, 3);

    search_dr_ion_p.create(nat, 3);
    search_dr_cell_p.create(3, 3);

    // set if we are allowing lattice vectors to move
    if_cell_moves = false;
    if (PARAM.inp.calculation == "cell-relax")
    {
        if_cell_moves = true;
    }
}

bool Relax::relax_step(UnitCell& ucell,
                       const ModuleBase::matrix& force,
                       const ModuleBase::matrix& stress,
                       const double etot_in)
{
    ModuleBase::TITLE("Relax", "relax_step");

    etot = etot_in * ModuleBase::Ry_to_eV; // convert to eV
    if (istep == 0)
    {
        etot_p = etot;
    }

    bool relax_done = this->setup_gradient(ucell, force, stress);
    if (relax_done)
    {
        return relax_done;
    }

    this->calculate_gamma();

    bool ls_done = this->check_line_search();

    if (ls_done)
    {
        this->new_direction();
        this->move_cell_ions(ucell, true);
    }
    else
    {
        this->perform_line_search();
        this->move_cell_ions(ucell, false);
        dmovel = dmove;
    }

    istep++;

    return relax_done;
}

bool Relax::setup_gradient(const UnitCell& ucell, const ModuleBase::matrix& force, const ModuleBase::matrix& stress)
{
    ModuleBase::TITLE("Relax", "setup_gradient");

    // if not relax, then return converged
    if (!(PARAM.inp.calculation == "relax" || PARAM.inp.calculation == "cell-relax"))
    {
        return true;
    }

    // indicating whether force & stress are converged
    bool force_converged = true;
    double max_grad = 0.0;

    //=========================================
    // set gradient for ions degrees of freedom
    //=========================================

    grad_ion.zero_out();
    ModuleBase::matrix force_eva = force * (ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A); // convert to eV/A

    int iat = 0;
    for (int it = 0; it < ucell.ntype; it++)
    {
        Atom* atom = &ucell.atoms[it];
        for (int ia = 0; ia < ucell.atoms[it].na; ia++)
        {
            double force2 = 0.0;
            if (atom->mbl[ia].x == 1)
            {
                grad_ion(iat, 0) = force_eva(iat, 0);
                if (std::abs(force_eva(iat, 0)) > max_grad)
                {
                    max_grad = std::abs(force_eva(iat, 0));
                }
            }
            if (atom->mbl[ia].y == 1)
            {
                grad_ion(iat, 1) = force_eva(iat, 1);
                if (std::abs(force_eva(iat, 1)) > max_grad)
                {
                    max_grad = std::abs(force_eva(iat, 1));
                }
            }
            if (atom->mbl[ia].z == 1)
            {
                grad_ion(iat, 2) = force_eva(iat, 2);
                if (std::abs(force_eva(iat, 2)) > max_grad)
                {
                    max_grad = std::abs(force_eva(iat, 2));
                }
            }
            ++iat;
        }
    }
    assert(iat == nat);

    if (max_grad > force_thr_eva)
    {
        force_converged = false;
    }
    if (PARAM.inp.out_level == "ie")
    {
        std::cout << " ETOT DIFF (eV)       : " << etot - etot_p << std::endl;
        std::cout << " LARGEST GRAD (eV/A)  : " << max_grad << std::endl;
        etot_p = etot;
    }

    GlobalV::ofs_running << "\n Largest gradient in force is " << max_grad << " eV/A." << std::endl;
    GlobalV::ofs_running << " Threshold is " << PARAM.inp.force_thr_ev << " eV/A." << std::endl;
    //=========================================
    // set gradient for cell degrees of freedom
    //=========================================

    if (if_cell_moves)
    {
        grad_cell.zero_out();
        ModuleBase::matrix stress_ev = stress * (ucell.omega * ModuleBase::Ry_to_eV);

        if (PARAM.inp.fixed_axes == "shape")
        {
            double pressure = (stress_ev(0, 0) + stress_ev(1, 1) + stress_ev(2, 2)) / 3.0;
            stress_ev.zero_out();
            stress_ev(0, 0) = pressure; // apply constraints
            stress_ev(1, 1) = pressure;
            stress_ev(2, 2) = pressure;
        }
        else if (PARAM.inp.fixed_axes == "volume")
        {
            double pressure = (stress_ev(0, 0) + stress_ev(1, 1) + stress_ev(2, 2)) / 3.0;
            stress_ev(0, 0) -= pressure;
            stress_ev(1, 1) -= pressure;
            stress_ev(2, 2) -= pressure;
        }
        else if (PARAM.inp.fixed_axes != "None")
        {
            // Note stress is given in the directions of lattice vectors
            // So we need to first convert to Cartesian and then apply the constraint
            ModuleBase::Matrix3 stress_cart;
            stress_cart.e11 = stress_ev(0, 0);
            stress_cart.e12 = stress_ev(0, 1);
            stress_cart.e13 = stress_ev(0, 2);
            stress_cart.e21 = stress_ev(1, 0);
            stress_cart.e22 = stress_ev(1, 1);
            stress_cart.e23 = stress_ev(1, 2);
            stress_cart.e31 = stress_ev(2, 0);
            stress_cart.e32 = stress_ev(2, 1);
            stress_cart.e33 = stress_ev(2, 2);

            stress_cart = ucell.latvec * stress_cart;

            if (ucell.lc[0] == 0)
            {
                stress_cart.e11 = 0;
                stress_cart.e12 = 0;
                stress_cart.e13 = 0;
            }
            if (ucell.lc[1] == 0)
            {
                stress_cart.e21 = 0;
                stress_cart.e22 = 0;
                stress_cart.e23 = 0;
            }
            if (ucell.lc[2] == 0)
            {
                stress_cart.e31 = 0;
                stress_cart.e32 = 0;
                stress_cart.e33 = 0;
            }

            stress_cart = ucell.GT * stress_cart;
            stress_ev(0, 0) = stress_cart.e11;
            stress_ev(0, 1) = stress_cart.e12;
            stress_ev(0, 2) = stress_cart.e13;
            stress_ev(1, 0) = stress_cart.e21;
            stress_ev(1, 1) = stress_cart.e22;
            stress_ev(1, 2) = stress_cart.e23;
            stress_ev(2, 0) = stress_cart.e31;
            stress_ev(2, 1) = stress_cart.e32;
            stress_ev(2, 2) = stress_cart.e33;
        }

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                grad_cell(i, j) = stress_ev(i, j); // apply constraints
            }
        }

        double largest_grad = 0.0;
        double stress_ii_max = 0.0;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double grad = grad_cell(i, j) / (ucell.omega * ModuleBase::Ry_to_eV);
                if (largest_grad < std::abs(grad))
                {
                    largest_grad = std::abs(grad);
                }
            }
        }

        double unit_transform = ModuleBase::RYDBERG_SI / pow(ModuleBase::BOHR_RADIUS_SI, 3) * 1.0e-8;
        largest_grad = largest_grad * unit_transform;

        if (largest_grad > PARAM.inp.stress_thr)
        {
            force_converged = false;
        }

        GlobalV::ofs_running << "\n Largest gradient in stress is " << largest_grad << " kbar." << std::endl;
        GlobalV::ofs_running << " Threshold is " << PARAM.inp.stress_thr << " kbar." << std::endl;
    }

    if (force_converged)
    {
        GlobalV::ofs_running << "\n Relaxation is converged!" << std::endl;
    }
    else
    {
        GlobalV::ofs_running << "\n Relaxation is not converged yet!" << std::endl;
    }

    return force_converged;
}

void Relax::calculate_gamma()
{
    ModuleBase::TITLE("Relax", "calculate_gamma");

    // no need to calculate gamma if last step is trial
    // since we won't update search direction
    if (ltrial)
    {
        return;
    }

    grp_grp = 0.0; // grad_p*grad_p
    gr_grp = 0.0;  // grad  *grad_p
    gr_gr = 0.0;   // grad  *grad
    gr_sr = 0.0;   // grad  *search_dir

    for (int iat = 0; iat < nat; iat++)
    {
        for (int i = 0; i < 3; i++)
        {
            grp_grp += grad_ion_p(iat, i) * grad_ion_p(iat, i);
            gr_grp += grad_ion_p(iat, i) * grad_ion(iat, i);
            gr_gr += grad_ion(iat, i) * grad_ion(iat, i);
            gr_sr += grad_ion(iat, i) * search_dr_ion(iat, i);
        }
    }

    if (if_cell_moves)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                grp_grp += grad_cell_p(i, j) * grad_cell_p(i, j) / nat;
                gr_grp += grad_cell_p(i, j) * grad_cell(i, j) / nat;
                gr_gr += grad_cell(i, j) * grad_cell(i, j) / nat;
                gr_sr += grad_cell(i, j) * search_dr_cell(i, j) / nat;
            }
        }
    }

    if (cg_step == 0)
    {
        gamma = 0.0;
    }
    else
    {
        gamma = (gr_gr - gr_grp) / grp_grp; // Polak-Riebere
    }
}

bool Relax::check_line_search()
{
    ModuleBase::TITLE("Relax", "check_line_search");

    // if last step is trial step towards new direction
    // then line search is not finished
    // we will perform line search
    if (ltrial)
    {
        ltrial = false;
        return false;
    }

    if (std::abs(gr_sr) * std::max(gamma, 1.0) > std::abs(gr_gr) / 5.0
        && !brent_done) // last brent line search not finished
    {
        return false;
    }

    return true;
}

void Relax::perform_line_search()
{
    ModuleBase::TITLE("Relax", "line_search");

    double f = 0.0; // 1st order energy difference
    for (int iat = 0; iat < nat; iat++)
    {
        for (int i = 0; i < 3; i++)
        {
            f -= step_size * fac_force * search_dr_ion(iat, i) * grad_ion(iat, i);
        }
    }

    if (if_cell_moves)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                f -= step_size * fac_stress * search_dr_cell(i, j) * grad_cell(i, j);
            }
        }
    }

    // perform line search
    bool restart_brent = false;
    double x = dmovel, y = etot;
    double xnew, yd;

    brent_done = this->ls.line_search(restart_brent, x, y, f, xnew, force_thr_eva);
    dmove = xnew;

    return;
}

void Relax::new_direction()
{
    ModuleBase::TITLE("Relax", "new_direction");
    if (cg_step != 0)
    {
        step_size += 0.2 * step_size * (dmovel - 1.0);
    }

    // set GAMMA to zero if line minimization was not sufficient
    if (5.0 * std::abs(gr_sr) * gamma > std::abs(gr_gr))
    {
        gamma = 0.0;
        cg_step = 0; // reset cg
    }

    // perform trial step

    // set search vectors
    search_dr_ion_p = search_dr_ion;
    search_dr_cell_p = search_dr_cell;

    grad_ion_p = grad_ion;
    grad_cell_p = grad_cell;

    search_dr_ion = grad_ion + gamma * search_dr_ion;
    search_dr_cell = grad_cell + gamma * search_dr_cell;

    // modify step if necessary
    sr_sr = 1.0e-10;
    for (int iat = 0; iat < nat; iat++)
    {
        for (int i = 0; i < 3; i++)
        {
            sr_sr += 1 / fac_force * search_dr_ion(iat, i) * search_dr_ion(iat, i);
        }
    }

    if (if_cell_moves)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                sr_sr += 1 / fac_stress * search_dr_cell(i, j) * search_dr_cell(i, j);
            }
        }
    }

    // if length of search vector increased, rescale step to avoid too large trial steps
    if (sr_sr > srp_srp)
    {
        step_size *= srp_srp / sr_sr;
    }
    srp_srp = sr_sr;

    double f = 0.0; // first order change in energy (gradient in the search direction)
    for (int iat = 0; iat < nat; iat++)
    {
        for (int i = 0; i < 3; i++)
        {
            f -= step_size * fac_force * search_dr_ion(iat, i) * grad_ion(iat, i);
        }
    }

    if (if_cell_moves)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                f -= step_size * fac_stress * search_dr_cell(i, j) * grad_cell(i, j);
            }
        }
    }

    // prepare for line search
    bool restart = true;
    double x = 0, y = etot;
    // TODO: add a certain threshold for the progress.
    double xnew, yd = 1e-8;

    this->ls.line_search(restart, x, y, f, xnew, yd);

    dmovel = 1.0;
    ltrial = true;
    cg_step++;

    return;
}

void Relax::move_cell_ions(UnitCell& ucell, const bool is_new_dir)
{
    ModuleBase::TITLE("Relax", "move_cell_ions");

    // I'm keeping this only because we have to
    // be compatible with old code
    ucell.ionic_position_updated = true;
    if (if_cell_moves)
    {
        ucell.cell_parameter_updated = true;
    }

    // Depending on whether this is a first step along CG new direction
    // or a line search step, the treatment is slightly different
    // and the input variable is_new_dir is used to make the distinction

    double fac; // fac1 for force, fac2 for stress
    if (is_new_dir)
    {
        fac = 1.0;
    }
    else
    {
        fac = dmove - dmovel;
    }

    // The sequence of operations in this subroutine is as follows:
    // First of all, update latvec
    // Secondly, update direct coordinates of atoms
    // in this step we need to transform displacement from Cartesian to direct
    // coordinates using the OLD G (reciprocal lattice vectors)
    // Thirdly, in update_pos_taud, update Cartesian coordinates of atoms
    // in this step we are using the NEW latvec (lattice vectors)
    // Finally, update G, GT and other stuff, and print the new STRU, and update something for next SCF

    // =================================================================
    // Step 1 : updating latvec
    // =================================================================

    if (if_cell_moves)
    {
        // imo matrix3 class is not a very clever way to store 3*3 matrix ...
        ModuleBase::Matrix3 sr_dr_cell;
        auto cp_mat_to_mat3 = [&sr_dr_cell, this]() -> void {
            sr_dr_cell.e11 = search_dr_cell(0, 0);
            sr_dr_cell.e12 = search_dr_cell(0, 1);
            sr_dr_cell.e13 = search_dr_cell(0, 2);
            sr_dr_cell.e21 = search_dr_cell(1, 0);
            sr_dr_cell.e22 = search_dr_cell(1, 1);
            sr_dr_cell.e23 = search_dr_cell(1, 2);
            sr_dr_cell.e31 = search_dr_cell(2, 0);
            sr_dr_cell.e32 = search_dr_cell(2, 1);
            sr_dr_cell.e33 = search_dr_cell(2, 2);
        };
        cp_mat_to_mat3();

        if (ModuleSymmetry::Symmetry::symm_flag && ucell.symm.nrotk > 0)
        {
            search_dr_cell = sr_dr_cell.Transpose().to_matrix();
            ucell.symm.symmetrize_mat3(search_dr_cell, ucell.lat);
            cp_mat_to_mat3();
            sr_dr_cell = sr_dr_cell.Transpose();
        }

        // The logic here is as follows: a line search is a continuation
        // in the new direction; but ucell.latvec now is already
        // different from when the current CG step starts;
        // as a result, we need to save latvec at the beginning of
        // each CG step
        if (is_new_dir)
        {
            latvec_save = ucell.latvec;
        }

        ModuleBase::Matrix3 move_cell = latvec_save * sr_dr_cell;

        // should be close to 0, but set again to avoid numerical issues
        if (ucell.lc[0] == 0)
        {
            move_cell.e11 = 0;
            move_cell.e12 = 0;
            move_cell.e13 = 0;
        }
        if (ucell.lc[1] == 0)
        {
            move_cell.e21 = 0;
            move_cell.e22 = 0;
            move_cell.e23 = 0;
        }
        if (ucell.lc[2] == 0)
        {
            move_cell.e31 = 0;
            move_cell.e32 = 0;
            move_cell.e33 = 0;
        }
        ucell.latvec += move_cell * (step_size * fac * fac_stress);

        if (PARAM.inp.fixed_axes == "volume")
        {
            double omega_new = std::abs(ucell.latvec.Det()) * pow(ucell.lat0, 3);
            ucell.latvec *= pow(ucell.omega / omega_new, 1.0 / 3.0);
        }
        if (PARAM.inp.fixed_ibrav)
        {
            unitcell::remake_cell(ucell.lat);
        }
    }

    // =================================================================
    // Step 2 & 3 : update direct & Cartesian atomic positions
    // =================================================================

    // Calculating displacement in Cartesian coordinate (in Angstrom)
    double move_ion[nat * 3];
    ModuleBase::zeros(move_ion, nat * 3);

    for (int iat = 0; iat < nat; iat++)
    {
        // Cartesian coordinate
        // convert from Angstrom to unit of latvec (Bohr)
        ModuleBase::Vector3<double> move_ion_cart;
        move_ion_cart.x = step_size * fac_force * search_dr_ion(iat, 0) / ModuleBase::BOHR_TO_A / ucell.lat0;
        move_ion_cart.y = step_size * fac_force * search_dr_ion(iat, 1) / ModuleBase::BOHR_TO_A / ucell.lat0;
        move_ion_cart.z = step_size * fac_force * search_dr_ion(iat, 2) / ModuleBase::BOHR_TO_A / ucell.lat0;

        // convert to Direct coordinate
        // note here the old GT is used
        ModuleBase::Vector3<double> move_ion_dr = move_ion_cart * ucell.GT;

        int it = ucell.iat2it[iat];
        int ia = ucell.iat2ia[iat];
        Atom* atom = &ucell.atoms[it];

        if (atom->mbl[ia].x == 1)
        {
            move_ion[iat * 3] = move_ion_dr.x * fac;
        }
        if (atom->mbl[ia].y == 1)
        {
            move_ion[iat * 3 + 1] = move_ion_dr.y * fac;
        }
        if (atom->mbl[ia].z == 1)
        {
            move_ion[iat * 3 + 2] = move_ion_dr.z * fac;
        }
    }

    if (ModuleSymmetry::Symmetry::symm_flag && ucell.symm.all_mbl && ucell.symm.nrotk > 0)
    {
        ucell.symm.symmetrize_vec3_nat(move_ion);
    }

    unitcell::update_pos_taud(ucell.lat,move_ion,ucell.ntype,ucell.nat,ucell.atoms);

    // Print the structure file.
    unitcell::print_tau(ucell.atoms,ucell.Coordinate,ucell.ntype,ucell.lat0,GlobalV::ofs_running);

    // =================================================================
    // Step 4 : update G,GT and other stuff
    // =================================================================

    if (if_cell_moves)
    {
        ucell.a1.x = ucell.latvec.e11;
        ucell.a1.y = ucell.latvec.e12;
        ucell.a1.z = ucell.latvec.e13;
        ucell.a2.x = ucell.latvec.e21;
        ucell.a2.y = ucell.latvec.e22;
        ucell.a2.z = ucell.latvec.e23;
        ucell.a3.x = ucell.latvec.e31;
        ucell.a3.y = ucell.latvec.e32;
        ucell.a3.z = ucell.latvec.e33;

#ifdef __MPI
        // distribute lattice vectors.
        Parallel_Common::bcast_double(ucell.latvec.e11);
        Parallel_Common::bcast_double(ucell.latvec.e12);
        Parallel_Common::bcast_double(ucell.latvec.e13);
        Parallel_Common::bcast_double(ucell.latvec.e21);
        Parallel_Common::bcast_double(ucell.latvec.e22);
        Parallel_Common::bcast_double(ucell.latvec.e23);
        Parallel_Common::bcast_double(ucell.latvec.e31);
        Parallel_Common::bcast_double(ucell.latvec.e32);
        Parallel_Common::bcast_double(ucell.latvec.e33);

        // distribute lattice vectors.
        Parallel_Common::bcast_double(ucell.a1.x);
        Parallel_Common::bcast_double(ucell.a1.y);
        Parallel_Common::bcast_double(ucell.a1.z);
        Parallel_Common::bcast_double(ucell.a2.x);
        Parallel_Common::bcast_double(ucell.a2.y);
        Parallel_Common::bcast_double(ucell.a2.z);
        Parallel_Common::bcast_double(ucell.a3.x);
        Parallel_Common::bcast_double(ucell.a3.y);
        Parallel_Common::bcast_double(ucell.a3.z);
#endif

        ucell.omega = std::abs(ucell.latvec.Det()) * ucell.lat0 * ucell.lat0 * ucell.lat0;

        ucell.GT = ucell.latvec.Inverse();
        ucell.G = ucell.GT.Transpose();
        ucell.GGT = ucell.G * ucell.GT;
        ucell.invGGT = ucell.GGT.Inverse();
    }

    // =================================================================
    // Step 6 : prepare something for next SCF
    // =================================================================
    // I have a strong feeling that this part should be
    // at the beginning of the next step (namely 'beforescf'),
    // but before we have a better organized Esolver
    // I do not want to change it
    if (if_cell_moves)
    {
        unitcell::setup_cell_after_vc(ucell,GlobalV::ofs_running);
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "SETUP UNITCELL");
    }
}

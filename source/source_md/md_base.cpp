#include "md_base.h"
#include "md_func.h"
#ifdef __MPI
#include "mpi.h"
#endif
#include "source_io/print_info.h"
#include "source_cell/update_cell.h"
MD_base::MD_base(const Parameter& param_in, UnitCell& unit_in) : mdp(param_in.mdp), ucell(unit_in)
{
    my_rank = param_in.globalv.myrank;
    cal_stress = param_in.inp.cal_stress;
    if (mdp.md_seed >= 0)
    {
        srand(mdp.md_seed);
    }

    stop = false;

    assert(ucell.nat>0);

    allmass = new double[ucell.nat];
    pos = new ModuleBase::Vector3<double>[ucell.nat];
    vel = new ModuleBase::Vector3<double>[ucell.nat];
    ionmbl = new ModuleBase::Vector3<int>[ucell.nat];
    force = new ModuleBase::Vector3<double>[ucell.nat];
    virial.create(3, 3);
    stress.create(3, 3);

    assert(ModuleBase::AU_to_FS!=0.0);
    assert(ModuleBase::Hartree_to_K!=0.0);

    /// convert to a.u. unit
    md_dt = mdp.md_dt / ModuleBase::AU_to_FS;
    md_tfirst = mdp.md_tfirst / ModuleBase::Hartree_to_K;
    md_tlast = mdp.md_tlast / ModuleBase::Hartree_to_K;

    step_ = 0;
    step_rst_ = 0;

    MD_func::init_vel(ucell, my_rank, mdp.md_restart, md_tfirst, allmass, frozen_freedom_, ionmbl, vel);
    t_current = MD_func::current_temp(kinetic, ucell.nat, frozen_freedom_, allmass, vel);
}


MD_base::~MD_base()
{
    delete[] allmass;
    delete[] pos;
    delete[] vel;
    delete[] ionmbl;
    delete[] force;
}


void MD_base::setup(ModuleESolver::ESolver* p_esolver, const std::string& global_readin_dir)
{
    if (mdp.md_restart)
    {
        restart(global_readin_dir);
    }

    ModuleIO::print_screen(0, 0, step_ + step_rst_);

    MD_func::force_virial(p_esolver, step_, ucell, potential, force, cal_stress, virial);
    MD_func::compute_stress(ucell, vel, allmass, cal_stress, virial, stress);
    ucell.ionic_position_updated = true;

    return;
}


void MD_base::first_half(std::ofstream& ofs)
{
    update_vel(force);
    update_pos();

    return;
}


void MD_base::second_half()
{
    update_vel(force);

    return;
}


void MD_base::update_pos()
{
    if (my_rank == 0)
    {
        for (int i = 0; i < ucell.nat; ++i)
        {
            for (int k = 0; k < 3; ++k)
            {
                if (ionmbl[i][k])
                {
                    pos[i][k] = vel[i][k] * md_dt / ucell.lat0;
                }
                else
                {
                    pos[i][k] = 0;
                }
            }
            pos[i] = pos[i] * ucell.GT;
        }
    }

#ifdef __MPI
    MPI_Bcast(pos, ucell.nat * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    unitcell::update_pos_taud(ucell.lat,pos,ucell.ntype,ucell.nat,ucell.atoms);

    return;
}


void MD_base::update_vel(const ModuleBase::Vector3<double>* force)
{
    if (my_rank == 0)
    {
        for (int i = 0; i < ucell.nat; ++i)
        {
            for (int k = 0; k < 3; ++k)
            {
                if (ionmbl[i][k])
                {
                    vel[i][k] += 0.5 * force[i][k] * md_dt / allmass[i];
                }
            }
        }
    }

#ifdef __MPI
    MPI_Bcast(vel, ucell.nat * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    return;
}


void MD_base::print_md(std::ofstream& ofs, const bool& cal_stress)
{
    if (my_rank!=0)
    {
        return;
    }

    t_current = MD_func::current_temp(kinetic, ucell.nat, frozen_freedom_, allmass, vel);

    assert(ModuleBase::BOHR_RADIUS_SI>0.0);

    const double unit_transform = ModuleBase::HARTREE_SI / pow(ModuleBase::BOHR_RADIUS_SI, 3) * 1.0e-8;
    double press = 0.0;
    for (int i = 0; i < 3; i++)
    {
        press += stress(i, i) / 3;
    }

    // screen output
    std::cout << " ------------------------------------------------------------------------------------------------"
              << std::endl;
    std::cout << " " << std::left << std::setw(20) << "Energy (Ry)" << std::left << std::setw(20) << "Potential (Ry)"
              << std::left << std::setw(20) << "Kinetic (Ry)" << std::left << std::setw(20) << "Temperature (K)";

    if (cal_stress)
    {
        std::cout << std::left << std::setw(20) << "Pressure (kbar)";
    }

    std::cout << std::endl;
    std::cout << " " << std::left << std::setw(20) << 2 * (potential + kinetic) << std::left << std::setw(20)
              << 2 * potential << std::left << std::setw(20) << 2 * kinetic << std::left << std::setw(20)
              << t_current * ModuleBase::Hartree_to_K;

    if (cal_stress)
    {
        std::cout << std::left << std::setw(20) << press * unit_transform;
    }

    std::cout << std::endl;
    std::cout << " ------------------------------------------------------------------------------------------------"
              << std::endl;

    // running_log output
    ofs.unsetf(std::ios::fixed);
    ofs << std::setprecision(8);
    ofs << " ------------------------------------------------------------------------------------------------"
        << std::endl;
    ofs << " " << std::left << std::setw(20) << "Energy (Ry)" << std::left << std::setw(20) << "Potential (Ry)"
        << std::left << std::setw(20) << "Kinetic (Ry)" << std::left << std::setw(20) << "Temperature (K)";

    if (cal_stress)
    {
        ofs << std::left << std::setw(20) << "Pressure (kbar)";
    }

    ofs << std::endl;
    ofs << " " << std::left << std::setw(20) << 2 * (potential + kinetic) << std::left << std::setw(20) << 2 * potential
        << std::left << std::setw(20) << 2 * kinetic << std::left << std::setw(20)
        << t_current * ModuleBase::Hartree_to_K;

    if (cal_stress)
    {
        ofs << std::left << std::setw(20) << press * unit_transform;
    }

    ofs << std::endl;
    ofs << " ------------------------------------------------------------------------------------------------"
        << std::endl;

    if (cal_stress)
    {
        MD_func::print_stress(ofs, virial, stress);
    }

    return;
}


void MD_base::write_restart(const std::string& global_out_dir)
{
    if (!my_rank)
    {
        std::stringstream ssc;
        ssc << global_out_dir << "Restart_md.dat";
        std::ofstream file(ssc.str().c_str());

        file << step_ + step_rst_ << std::endl;
        file << md_tfirst << std::endl;
        file.close();
    }
#ifdef __MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    return;
}


void MD_base::restart(const std::string& global_readin_dir)
{
    MD_func::current_md_info(my_rank, global_readin_dir, step_rst_, md_tfirst);

    return;
}

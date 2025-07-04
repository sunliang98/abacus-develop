#include <vector>

#include "charge.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "module_parameter/parameter.h"
#include "source_base/libm/libm.h"
#include "source_base/math_integral.h"
#include "source_base/math_sphbes.h"
#include "source_base/memory.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/tool_threading.h"
#include "source_estate/magnetism.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "source_pw/hamilt_pwdft/parallel_grid.h"
#include "source_io/cube_io.h"
#include "source_io/rhog_io.h"
#include "source_io/read_wf2rho_pw.h"

void Charge::init_rho(elecstate::efermi& eferm_iout,
                      const UnitCell& ucell,
                      const Parallel_Grid& pgrid,
                      const ModuleBase::ComplexMatrix& strucFac,
                      ModuleSymmetry::Symmetry& symm,
                      const void* klist,
                      const void* wfcpw)
{
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "init_chg", PARAM.inp.init_chg);

    const int nspin = PARAM.inp.nspin;
    assert(nspin>0);

    std::cout << " START CHARGE      : " << PARAM.inp.init_chg << std::endl;

    // we need to set the omega for the charge density
    set_omega(&ucell.omega);
    this->pgrid = &pgrid;

    bool read_error = false;
    bool read_kin_error = false;
    if (PARAM.inp.init_chg == "file" || PARAM.inp.init_chg == "auto")
    {
        GlobalV::ofs_running << " Read electron density from file" << std::endl;

        // try to read charge from binary file first, which is the same as QE
        // liuyu 2023-12-05
        std::stringstream binary;
        binary << PARAM.globalv.global_readin_dir << PARAM.inp.suffix + "-CHARGE-DENSITY.restart";
        if (ModuleIO::read_rhog(binary.str(), rhopw, rhog))
        {
            GlobalV::ofs_running << " Read electron density from file: " << binary.str() << std::endl;
            for (int is = 0; is < nspin; ++is)
            {
                rhopw->recip2real(rhog[is], rho[is]);
            }
        }
        else
        {
            for (int is = 0; is < nspin; ++is)
            {
                std::stringstream ssc;
                ssc << PARAM.globalv.global_readin_dir << "chgs" << is + 1 << ".cube";
                if (ModuleIO::read_vdata_palgrid(pgrid,
                    (PARAM.inp.esolver_type == "sdft" ? GlobalV::RANK_IN_BPGROUP : GlobalV::MY_RANK),
                    GlobalV::ofs_running,
                    ssc.str(),
                    this->rho[is],
                    ucell.nat))
                {
                    GlobalV::ofs_running << " Read electron density from file: " << ssc.str() << std::endl;
                }
                else if (is > 0)    // nspin=2 or 4
                {
                    if (is == 1)    // failed at the second spin
                    {
                        std::cout << " Incomplete electron density file." << std::endl;
                        read_error = true;
                        break;
                    }
                    else if (is == 2)   // read 2 files when nspin=4
                    {
                        GlobalV::ofs_running << " Didn't read in the electron density but would rearrange it later. "
                            << std::endl;
                    }
                    else if (is == 3)   // read 2 files when nspin=4
                    {
                        GlobalV::ofs_running << " rearrange electron density " << std::endl;
                        for (int ir = 0; ir < this->rhopw->nrxx; ir++)
                        {
                            this->rho[3][ir] = this->rho[0][ir] - this->rho[1][ir];
                            this->rho[0][ir] = this->rho[0][ir] + this->rho[1][ir];
                            this->rho[1][ir] = 0.0;
                            this->rho[2][ir] = 0.0;
                        }
                    }
                }
                else
                {
                    read_error = true;
                    break;
                }
            }
        }

        if (read_error)
        {
            const std::string warn_msg
                = " WARNING: \"init_chg\" is enabled but ABACUS failed to read\n charge density from file.\n"
                  " Please check if there is chgsx.cube (x=1,2,etc.) or\n {suffix}-CHARGE-DENSITY.restart in the "
                  "directory.\n";
            std::cout << warn_msg;
            if (PARAM.inp.init_chg == "file")
            {
                ModuleBase::WARNING_QUIT("Charge::init_rho",
                                         "Failed to read in charge density from file.\n For initializing atomic "
                                         "charge in calculations,\n please set init_chg to atomic in INPUT.");
            }
        }

        if (XC_Functional::get_ked_flag())
        {
            // If the charge density is not read in, then the kinetic energy density is not read in either
            if (!read_error)
            {
                GlobalV::ofs_running << " try to read kinetic energy density from file" << std::endl;
                // try to read charge from binary file first, which is the same as QE
                std::vector<std::complex<double>> kin_g_space(nspin * this->ngmc, {0.0, 0.0});
                std::vector<std::complex<double>*> kin_g;
                for (int is = 0; is < nspin; is++)
                {
                    kin_g.push_back(kin_g_space.data() + is * this->ngmc);
                }

                std::stringstream binary;
                binary << PARAM.globalv.global_readin_dir << PARAM.inp.suffix + "-TAU-DENSITY.restart";
                if (ModuleIO::read_rhog(binary.str(), rhopw, kin_g.data()))
                {
                    GlobalV::ofs_running << " Read in the kinetic energy density: " << binary.str() << std::endl;
                    for (int is = 0; is < nspin; ++is)
                    {
                        rhopw->recip2real(kin_g[is], this->kin_r[is]);
                    }
                }
                else
                {
                    for (int is = 0; is < nspin; is++)
                    {
                        std::stringstream ssc;
                        ssc << PARAM.globalv.global_readin_dir << "SPIN" << is + 1 << "_TAU.cube";
                        // mohan update 2012-02-10, sunliang update 2023-03-09
                        if (ModuleIO::read_vdata_palgrid(
                                pgrid,
                                (PARAM.inp.esolver_type == "sdft" ? GlobalV::RANK_IN_BPGROUP : GlobalV::MY_RANK),
                                GlobalV::ofs_running,
                                ssc.str(),
                                this->kin_r[is],
                                ucell.nat))
                        {
                            GlobalV::ofs_running << " Read in the kinetic energy density: " << ssc.str() << std::endl;
                        }
                        else
                        {
                            read_kin_error = true;
                            std::cout << " WARNING: \"init_chg\" is enabled but ABACUS failed to read kinetic energy "
                                         "density from file.\n"
                                         " Please check if there is SPINX_TAU.cube (X=1,...) or "
                                         "{suffix}-TAU-DENSITY.restart in the directory.\n"
                                      << std::endl;
                            break;
                        }
                    }
                }
            }
            else
            {
                read_kin_error = true;
            }
        }
    }

    if (PARAM.inp.init_chg == "atomic" || read_error)
    {
        if (read_error)
        {
            std::cout << " Charge::init_rho: use atomic initialization instead." << std::endl;
        }
        this->atomic_rho(nspin, ucell.omega, rho, strucFac, ucell);
    }

    // initial tau = 3/5 rho^2/3, Thomas-Fermi
    if (XC_Functional::get_ked_flag())
    {
        if (PARAM.inp.init_chg == "atomic" || read_kin_error)
        {
            if (read_kin_error)
            {
                std::cout << " Charge::init_rho: init kinetic energy density from rho." << std::endl;
            }
            const double fact = (3.0 / 5.0) * pow(3.0 * ModuleBase::PI * ModuleBase::PI, 2.0 / 3.0);
            for (int is = 0; is < nspin; ++is)
            {
                for (int ir = 0; ir < this->rhopw->nrxx; ++ir)
                {
                    kin_r[is][ir] = fact * pow(std::abs(rho[is][ir]) * nspin, 5.0 / 3.0) / nspin;
                }
            }
        }
    }

    // Peize Lin add 2020.04.04
    if (GlobalC::restart.info_load.load_charge && !GlobalC::restart.info_load.load_charge_finish)
    {
        for (int is = 0; is < nspin; ++is)
        {
            try
            {
                GlobalC::restart.load_disk("charge", is, this->nrxx, rho[is]);
            }
            catch (const std::exception& e)
            {
                // try to load from the output of `out_chg` 
                std::stringstream ssc;
                ssc << PARAM.globalv.global_readin_dir << "chgs" << is + 1 << ".cube";
                if (ModuleIO::read_vdata_palgrid(pgrid,
                    (PARAM.inp.esolver_type == "sdft" ? GlobalV::RANK_IN_BPGROUP : GlobalV::MY_RANK),
                    GlobalV::ofs_running,
                    ssc.str(),
                    this->rho[is],
                    ucell.nat))
                {
                    GlobalV::ofs_running << " Read in electron density: " << ssc.str() << std::endl;
                }
            }
        }
        GlobalC::restart.info_load.load_charge_finish = true;
    }

#ifdef __MPI
    this->init_chgmpi();
#endif
    if (PARAM.inp.init_chg == "wfc")
    {
        if (wfcpw == nullptr)
        {
            ModuleBase::WARNING_QUIT("Charge::init_rho", "wfc is only supported for PW-KSDFT.");
        }

        const ModulePW::PW_Basis_K* pw_wfc = reinterpret_cast<ModulePW::PW_Basis_K*>(const_cast<void*>(wfcpw));
        const K_Vectors* kv = reinterpret_cast<const K_Vectors*>(klist);

		ModuleIO::read_wf2rho_pw(pw_wfc, symm, *this,
                PARAM.globalv.global_readin_dir,
				GlobalV::KPAR, GlobalV::MY_POOL, GlobalV::MY_RANK, 
                GlobalV::NPROC_IN_POOL, GlobalV::RANK_IN_POOL,
				PARAM.inp.nbands, nspin, PARAM.globalv.npol,
				kv->get_nkstot(),kv->ik2iktot,kv->isk,GlobalV::ofs_running);
    }
}

//==========================================================
// computes the core charge on the real space 3D mesh.
//==========================================================
void Charge::set_rho_core(const UnitCell& ucell,
                          const ModuleBase::ComplexMatrix& structure_factor, 
                          const bool* numeric)
{
    ModuleBase::TITLE("Charge","set_rho_core");
    ModuleBase::timer::tick("Charge","set_rho_core");

    bool bl = false;
    for (int it = 0; it<ucell.ntype; it++)
    {
        if (ucell.atoms[it].ncpp.nlcc)
        {
            bl = true;
            break;
        }
    }

    if (!bl)
    {
        ModuleBase::GlobalFunc::ZEROS( this->rho_core, this->rhopw->nrxx);
    	ModuleBase::timer::tick("Charge","set_rho_core");
        return;
    }

    double *rhocg = new double[this->rhopw->ngg];
    ModuleBase::GlobalFunc::ZEROS(rhocg, this->rhopw->ngg );

	// three dimension.
    std::complex<double> *vg = new std::complex<double>[this->rhopw->npw];	

    for (int it = 0; it < ucell.ntype;it++)
    {
        if (ucell.atoms[it].ncpp.nlcc)
        {
//----------------------------------------------------------
// EXPLAIN : drhoc compute the radial fourier transform for
// each shell of g vec
//----------------------------------------------------------
            this->non_linear_core_correction(
                numeric,
                ucell.omega,
                ucell.tpiba2,
                ucell.atoms[it].ncpp.msh,
                ucell.atoms[it].ncpp.r.data(),
                ucell.atoms[it].ncpp.rab.data(),
                ucell.atoms[it].ncpp.rho_atc.data(),
                rhocg);
//----------------------------------------------------------
// EXPLAIN : multiply by the structure factor and sum
//----------------------------------------------------------
            for (int ig = 0; ig < this->rhopw->npw ; ig++)
            {
                vg[ig] += structure_factor(it, ig) * rhocg[this->rhopw->ig2igg[ig]];
            }
        }
    }

	// for tmp use.
	for(int ig=0; ig< this->rhopw->npw; ig++)
	{
		this->rhog_core[ig] = vg[ig];
	}

    this->rhopw->recip2real(vg, this->rho_core);

    // test on the charge and computation of the core energy
    double rhoima = 0.0;
    double rhoneg = 0.0;
    for (int ir = 0; ir < this->rhopw->nrxx; ir++)
    {
        rhoneg += std::min(0.0, this->rhopw->fft_bundle.get_auxr_data<double>()[ir].real());
        rhoima += std::abs(this->rhopw->fft_bundle.get_auxr_data<double>()[ir].imag());
        // NOTE: Core charge is computed in reciprocal space and brought to real
        // space by FFT. For non smooth core charges (or insufficient cut-off)
        // this may result in negative values in some grid points.
        // Up to October 1999 the core charge was forced to be positive definite.
        // This induces an error in the force, and probably stress, calculation if
        // the number of grid points where the core charge would be otherwise neg
        // is large. The error disappears for sufficiently high cut-off, but may be
        // rather large and it is better to leave the core charge as it is.
        // If you insist to have it positive definite (with the possible problems
        // mentioned above) uncomment the following lines.  SdG, Oct 15 1999
    }

	// mohan fix bug 2011-04-03
    Parallel_Reduce::reduce_pool(rhoneg);
    Parallel_Reduce::reduce_pool(rhoima);

	// mohan changed 2010-2-2, make this same as in atomic_rho.
	// still lack something......
    rhoneg /= this->rhopw->nxyz * ucell.omega;
    rhoima /= this->rhopw->nxyz * ucell.omega;

    // calculate core_only exch-corr energy etxcc=E_xc[rho_core] if required
    // The term was present in previous versions of the code but it shouldn't
    delete [] rhocg;
    delete [] vg;
    ModuleBase::timer::tick("Charge","set_rho_core");
    return;
} // end subroutine set_rhoc

void Charge::set_rho_core_paw()
{
    ModuleBase::TITLE("Charge","set_rho_core_paw");
}


void Charge::non_linear_core_correction
(
    const bool &numeric,
    const double omega,
    const double tpiba2,
    const int mesh,
    const double *r,
    const double *rab,
    const double *rhoc,
    double *rhocg) const
{
    ModuleBase::TITLE("charge","drhoc");

	// use labmda instead of repeating codes
	const auto kernel = [&](int num_threads, int thread_id)
	{

	double gx = 0.0;
    double rhocg1 = 0.0;
    double *aux;

    // here we compute the fourier transform is the charge in numeric form
    if (numeric)
    {
        aux = new double [mesh];
        // G=0 term

        int igl0 = 0;
        if (this->rhopw->gg_uniq [0] < 1.0e-8)
        {
			// single thread term
			if (thread_id == 0)
			{
				for (int ir = 0;ir < mesh; ir++)
				{
					aux [ir] = r [ir] * r [ir] * rhoc [ir];
				}
				ModuleBase::Integral::Simpson_Integral(mesh, aux, rab, rhocg1);
				//rhocg [1] = fpi * rhocg1 / omega;
				rhocg [0] = ModuleBase::FOUR_PI * rhocg1 / omega;//mohan modify 2008-01-19
			}
            igl0 = 1;
        }

		int igl_beg, igl_end;
		// exclude igl0
		ModuleBase::TASK_DIST_1D(num_threads, thread_id, this->rhopw->ngg - igl0, igl_beg, igl_end);
		igl_beg += igl0;
		igl_end += igl_beg;

        // G <> 0 term
        for (int igl = igl_beg; igl < igl_end;igl++) 
        {
            gx = sqrt(this->rhopw->gg_uniq[igl] * tpiba2);
            ModuleBase::Sphbes::Spherical_Bessel(mesh, r, gx, 0, aux);
            for (int ir = 0;ir < mesh; ir++) 
            {
                aux [ir] = r[ir] * r[ir] * rhoc [ir] * aux [ir];
            } //  enddo
            ModuleBase::Integral::Simpson_Integral(mesh, aux, rab, rhocg1);
            rhocg [igl] = ModuleBase::FOUR_PI * rhocg1 / omega;
        } //  enddo
        delete [] aux;
    }
    else
    {
        // here the case where the charge is in analytic form,
        // check old version before 2008-12-9
    }

	}; // end kernel

	// do not use omp parallel when this function is already in parallel block
	// 
	// it is called in parallel block in Forces::cal_force_cc,
	// but not in other funtcion such as Stress_Func::stress_cc.
	ModuleBase::TRY_OMP_PARALLEL(kernel);

    return;
}

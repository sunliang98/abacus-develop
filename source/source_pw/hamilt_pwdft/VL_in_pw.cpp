#include "VL_in_pw.h"
#include "module_parameter/parameter.h"
#include "source_base/libm/libm.h"
#include "source_base/math_integral.h"
#include "source_base/timer.h"
#include "source_pw/hamilt_pwdft/global.h"

pseudopot_cell_vl::pseudopot_cell_vl()
{
	numeric = nullptr;
	zp = nullptr; 
}

pseudopot_cell_vl::~pseudopot_cell_vl()
{
	delete[] numeric;
	delete[] zp;
}

void pseudopot_cell_vl::init_vloc(const UnitCell& ucell,
								  const ModulePW::PW_Basis* rho_basis)
{

	ModuleBase::TITLE("pseudopot_cell_vl","init_vloc");

	// This routine computes the fourier coefficient of the local
	// potential vloc(ig,it) for each type of atom
	ModuleBase::timer::tick("ppcell_vl","init_vloc");

	double *vloc1d = new double[rho_basis->ngg];
	ModuleBase::GlobalFunc::ZEROS(vloc1d, rho_basis->ngg);

	this->allocate(ucell,rho_basis->ngg);
	
	for (int it = 0; it < ucell.ntype; it++) 
	{
		const Atom* atom = &ucell.atoms[it];

		ModuleBase::GlobalFunc::ZEROS(vloc1d, rho_basis->ngg);

		this->zp[it] = atom->ncpp.zv;
		// compute V_loc(G) for a given type of atom
		if(atom->coulomb_potential)
		{
			this->vloc_coulomb(ucell,this->zp[it], vloc1d, rho_basis);
		}
		else if(numeric[it]==true)
		{
			this->vloc_of_g(
					atom->ncpp.msh, // after cutoff 
					atom->ncpp.rab.data(),
		          	atom->ncpp.r.data(), 
					atom->ncpp.vloc_at.data(), // local potential in real space radial form.  
		          	this->zp[it],
					vloc1d,
					ucell,
					rho_basis);
		}
		else
		{
			ModuleBase::WARNING_QUIT("init_vloc","not available now.");
		}

		if(it>=0 && it<this->vloc.nr && this->vloc.nc>=0)
		{
			ModuleBase::GlobalFunc::COPYARRAY(vloc1d, &this->vloc(it, 0), rho_basis->ngg);
		}
	} 


	delete[] vloc1d;

	this->print_vloc(ucell,rho_basis);

	ModuleBase::timer::tick("ppcell_vl","init_vloc");
	return;
}


void pseudopot_cell_vl::allocate(const UnitCell& ucell,
							     const int ngg)
{
	if(PARAM.inp.test_pp>0) 
	{ 
		ModuleBase::TITLE("pseudopot_cell_vl","allocate");
	}

	this->vloc.create(ucell.ntype, ngg);

	delete[] numeric;
	this->numeric = new bool[ucell.ntype];
	ModuleBase::GlobalFunc::ZEROS(numeric, ucell.ntype);

	for (int it = 0; it < ucell.ntype; it++)
	{ 
		this->numeric[it] = true; 
	}

	// mohan change global variable 'npsx' to local variable,
	// npsx( max number of different PPs)
	// 2021-02-22
	int npsx = 50;
	delete[] zp; 
	this->zp = new double[npsx];
	ModuleBase::GlobalFunc::ZEROS(zp, npsx);

	return;
}

void pseudopot_cell_vl::vloc_coulomb(const UnitCell& ucell,
									 const double& zp_in, 
									 double* vloc_1d, 
									 const ModulePW::PW_Basis* rho_basis) const
{
    int igl0 = 0;
    // start from |G|=0 or not.
	if (rho_basis->gg_uniq[0] < 1.0e-8)
    {
        igl0 = 1;
		vloc_1d[0] = 0.0;
    }
    else
    {
        igl0 = 0;
    }
    const double d_fpi_omega = ModuleBase::FOUR_PI / ucell.omega; // mohan add 2008-06-04
    double fac = -zp_in * ModuleBase::e2 * d_fpi_omega;
#ifdef _OPENMP
#pragma omp for
#endif
    for (int ig = igl0; ig < rho_basis->ngg; ig++)
    {
        double gx2 = rho_basis->gg_uniq[ig] * ucell.tpiba2;
        vloc_1d[ig] = fac / gx2;
    }
    return;
}

/*>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
!
!  This code transfer local pseudopotential in real space
!      radial logarithmic mesh to G space.
!
!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>*/


// Here we always have numeric form, i.e. numeric=ture
void pseudopot_cell_vl::vloc_of_g(const int& msh,
                                  const double* rab,
                                  const double* r,
                                  const double* vloc_at,
                                  const double& zp_in,
                                  double* vloc_1d,
								  const UnitCell& ucell,
                                  const ModulePW::PW_Basis* rho_basis) const
{
	//----------------------------------------------------------------
	//    This routine computes the Fourier transform of the local
	//    part of the pseudopotential. Two types of local potentials
	//    are allowed:

	//    a) The pseudopotential is in analytic form and its fourier
	//       transform is computed analytically
	//    b) The pseudopotential is in numeric form and its fourier
	//       transform is computed numerically
	//
	//    The local pseudopotential of the US case is always in
	//    numerical form, expressed in Ry units.
	// ----------------------------------------------------------------
	int igl0=0;// start from |G|=0 or not.

	// Pseudopotentials in numerical form (Vloc_at) contain the local part
	// in order to perform the Fourier transform, a term erf(r)/r is
	// subtracted in real space and added again in G space

    assert(msh>0);
	
	double *aux1 = new double[msh];

    // (1)
	if(rho_basis->gg_uniq[0] < 1.0e-8)
	{
		double *aux = new double[msh];
		// first the g=0 term
		for (int ir=0; ir<msh; ir++) 
		{
			// This is the |G| = 0 component of the local
			// potential giving rise to the so-called
			// "alpha*Z" term in the energy.
			aux[ir] = r [ir] * (r [ir] * vloc_at [ir] + zp_in * ModuleBase::e2);
			//aux[ir] = r [ir] * (r [ir] * vloc_at [ir] );
		}
		ModuleBase::Integral::Simpson_Integral(msh, aux, rab, vloc_1d[0] );
		igl0 = 1;	
		delete [] aux;
	}
	else
	{
		igl0 = 0;
	}

	// (2) here the |G|>0 terms, we first compute the part of the integrand func
	// indipendent of |G| in real space
	double fac = zp_in * ModuleBase::e2;
	for (int ir = 0;ir < msh;ir++)  
	{
		aux1 [ir] = r[ir] * vloc_at [ir] + fac * erf(r[ir]);
	} 

	// here we perform the integral, after multiplying for the |G|
	// dependent part
#ifdef _OPENMP
#pragma omp parallel
{
#endif
	double *aux = new double[msh];

#ifdef _OPENMP
#pragma omp for
#endif
	for (int ig = igl0;ig < rho_basis->ngg;ig++) 
	{
		double gx2= rho_basis->gg_uniq[ig] * ucell.tpiba2;
		double gx = std::sqrt(gx2);
		for (int ir = 0;ir < msh;ir++) 
		{
			aux [ir] = aux1 [ir] * ModuleBase::libm::sin(gx * r [ir]) / gx;
		}
		ModuleBase::Integral::Simpson_Integral(msh, aux, rab, vloc_1d[ig] );
		//  here we add the analytic fourier transform of the erf function
		vloc_1d[ig] -= fac * ModuleBase::libm::exp(- gx2 * 0.25)/ gx2;
	} // enddo

	const double d_fpi_omega = ModuleBase::FOUR_PI/ucell.omega;//mohan add 2008-06-04
#ifdef _OPENMP
#pragma omp for
#endif
	for (int ig = 0;ig < rho_basis->ngg; ig++)
	{
		vloc_1d[ig] *= d_fpi_omega;
	}

	delete [] aux;
#ifdef _OPENMP
}
#endif

	delete [] aux1;
	return;
} // end subroutine vloc_of_g


void pseudopot_cell_vl::print_vloc(const UnitCell& ucell,
								   const ModulePW::PW_Basis* rho_basis) const
{
	if(GlobalV::MY_RANK!=0) 
	{ 
		return; //mohan fix bug 2011-10-13
	}

	bool check_vl = PARAM.inp.out_element_info;

	if(check_vl)
	{
		for(int it=0; it<ucell.ntype; it++)
		{
			std::stringstream ss ;
			ss << PARAM.globalv.global_out_dir << ucell.atoms[it].label << "/v_loc_g.dat" ;
			std::ofstream ofs_vg( ss.str().c_str() );
			for(int ig=0;ig<rho_basis->ngg;ig++)
			{
				ofs_vg << std::setw(15) << rho_basis->gg_uniq [ig] * ucell.tpiba2 
				   	<< std::setw(15) << this->vloc(it, ig) << std::endl;
			}
			ofs_vg.close();
		}
	}
	return;
}

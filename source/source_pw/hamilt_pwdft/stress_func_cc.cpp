#include "stress_func.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "module_parameter/parameter.h"
#include "source_base/math_integral.h"
#include "source_base/timer.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "source_estate/cal_ux.h"

#ifdef USE_LIBXC
#include "source_hamilt/module_xc/xc_functional_libxc.h"
#endif


//NLCC term, need to be tested
template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::stress_cc(ModuleBase::matrix& sigma,
                                            ModulePW::PW_Basis* rho_basis,
											UnitCell& ucell,
                                            const Structure_Factor* p_sf,
                                            const bool is_pw,
											const bool *numeric,
                                            const Charge* const chr)
{
    ModuleBase::TITLE("Stress","stress_cc");
	ModuleBase::timer::tick("Stress","stress_cc");
        
	FPTYPE fact=1.0;

	if(is_pw&&PARAM.globalv.gamma_only_pw) 
	{
		fact = 2.0; //is_pw:PW basis, gamma_only need to FPTYPE.
	}

	FPTYPE sigmadiag;
	FPTYPE* rhocg;

	int judge=0;
	for(int nt=0;nt<ucell.ntype;nt++)
	{
		if(ucell.atoms[nt].ncpp.nlcc) 
		{
			judge++;
		}
	}

	if(judge==0) 
	{
		ModuleBase::timer::tick("Stress","stress_cc");
		return;
	}

	//recalculate the exchange-correlation potential
	ModuleBase::matrix vxc;
    if (XC_Functional::get_ked_flag())
    {
#ifdef USE_LIBXC
        const auto etxc_vtxc_v
            = XC_Functional_Libxc::v_xc_meta(XC_Functional::get_func_id(), rho_basis->nrxx, ucell.omega, ucell.tpiba, chr);

        // etxc = std::get<0>(etxc_vtxc_v);
        // vtxc = std::get<1>(etxc_vtxc_v);
        vxc = std::get<2>(etxc_vtxc_v);
#else
        ModuleBase::WARNING_QUIT("cal_force_cc","to use mGGA, compile with LIBXC");
#endif
	}
	else
	{
		elecstate::cal_ux(ucell);
        const auto etxc_vtxc_v = XC_Functional::v_xc(rho_basis->nrxx, chr, &ucell);
        // etxc = std::get<0>(etxc_vtxc_v); // may delete?
        // vtxc = std::get<1>(etxc_vtxc_v); // may delete?
        vxc = std::get<2>(etxc_vtxc_v);
    }

    std::complex<FPTYPE>* psic = new std::complex<FPTYPE>[rho_basis->nmaxgr];

    if(PARAM.inp.nspin==1||PARAM.inp.nspin==4)
	{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ir=0;ir<rho_basis->nrxx;ir++)
		{
			// psic[ir] = vxc(0,ir);
			psic[ir] = std::complex<FPTYPE>(vxc(0, ir),  0.0);
		}
	}
	else
	{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ir=0;ir<rho_basis->nrxx;ir++)
		{
			psic[ir] = 0.5 * (vxc(0, ir) + vxc(1, ir));
		}
	}

	// to G space
	rho_basis->real2recip(psic, psic); 

	//psic cantains now Vxc(G)
	rhocg= new FPTYPE [rho_basis->ngg];

	sigmadiag=0.0;
	for(int nt=0;nt<ucell.ntype;nt++)
	{
		if(ucell.atoms[nt].ncpp.nlcc)
		{
			//drhoc();
			this->deriv_drhoc(
				numeric,
				ucell.omega,
				ucell.tpiba2,
				ucell.atoms[nt].ncpp.msh,
				ucell.atoms[nt].ncpp.r.data(),
				ucell.atoms[nt].ncpp.rab.data(),
				ucell.atoms[nt].ncpp.rho_atc.data(),
				rhocg,
				rho_basis,
				1);


			//diagonal term 
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sigmadiag) schedule(static, 256)
#endif
			for(int ig = 0;ig< rho_basis->npw;ig++)
			{
                std::complex<double> local_sigmadiag;
                if (rho_basis->ig_gge0 == ig) {
                    local_sigmadiag = conj(psic[ig]) * p_sf->strucFac(nt, ig) * rhocg[rho_basis->ig2igg[ig]];
                } else {
                    local_sigmadiag = conj(psic[ig]) * p_sf->strucFac(nt, ig) * rhocg[rho_basis->ig2igg[ig]] * fact;
}
                sigmadiag += local_sigmadiag.real();
            }
			this->deriv_drhoc (
				numeric,
				ucell.omega,
				ucell.tpiba2,
				ucell.atoms[nt].ncpp.msh,
				ucell.atoms[nt].ncpp.r.data(),
				ucell.atoms[nt].ncpp.rab.data(),
				ucell.atoms[nt].ncpp.rho_atc.data(),
				rhocg,
				rho_basis,
				0);
			// non diagonal term (g=0 contribution missing)
#ifdef _OPENMP
#pragma omp parallel
{
			ModuleBase::matrix local_sigma(3, 3);
			#pragma omp for
#else
			ModuleBase::matrix& local_sigma = sigma;
#endif
			for(int ig = 0;ig< rho_basis->npw;ig++)
			{
				const FPTYPE norm_g = sqrt(rho_basis->gg[ig]);
				if(norm_g < 1e-4) { 	continue;
}
				for (int l = 0; l < 3; l++)
				{
					for (int m = 0;m< 3;m++)
					{
                        const std::complex<FPTYPE> t
                            = conj(psic[ig]) * p_sf->strucFac(nt, ig) * rhocg[rho_basis->ig2igg[ig]]
                              * ucell.tpiba * rho_basis->gcar[ig][l] * rho_basis->gcar[ig][m] / norm_g * fact;
                        //						sigmacc [l][ m] += t.real();
                        local_sigma(l,m) += t.real();
					}//end m
				}//end l
			}//end ng
#ifdef _OPENMP
			#pragma omp critical(stress_cc_reduce)
			{
				for(int l=0;l<3;l++)
				{
					for(int m=0;m<3;m++)
					{
						sigma(l,m) += local_sigma(l,m);
					}
				}
			}
}
#endif
		}//end if
	}//end nt

	for(int l = 0;l< 3;l++)
	{
		sigma(l,l) += sigmadiag;
	}
	for(int l = 0;l< 3;l++)
	{
		for (int m = 0;m< 3;m++)
		{
            Parallel_Reduce::reduce_pool(sigma(l, m));
		}
	}

	delete[] rhocg;
	delete[] psic;

	ModuleBase::timer::tick("Stress","stress_cc");
	return;
}


template<typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::deriv_drhoc
(
	const bool &numeric,
	const double& omega,
	const double& tpiba2,
	const int mesh,
	const FPTYPE *r,
	const FPTYPE *rab,
	const FPTYPE *rhoc,
	FPTYPE *drhocg,
	ModulePW::PW_Basis* rho_basis,
	int type
)
{
	int igl0=0;
	double gx = 0.0;
    double rhocg1 = 0.0;
	std::vector<double> aux(mesh);
	this->device = base_device::get_device_type<Device>(this->ctx);

	// the modulus of g for a given shell
	// the fourier transform
	// auxiliary memory for integration
	std::vector<double> gx_arr(rho_basis->ngg);
	double *gx_arr_d = nullptr;

	// counter on radial mesh points
	// counter on g shells
	// lower limit for loop on ngl

	//
	// G=0 term
	//
	if(type == 0)
	{
		if (rho_basis->gg_uniq[0] < 1.0e-8)
		{
			drhocg [0] = 0.0;
			igl0 = 1;
		}
		else
		{
			igl0 = 0;
		}
	} 
	else 
	{
		if (rho_basis->gg_uniq[0] < 1.0e-8)
		{
			for (int ir = 0;ir < mesh; ir++)
			{
				aux [ir] = r [ir] * r [ir] * rhoc [ir];
			}
			ModuleBase::Integral::Simpson_Integral(mesh, aux.data(), rab, rhocg1);
			drhocg [0] = ModuleBase::FOUR_PI * rhocg1 / omega;
			igl0 = 1;
		} 
		else
		{
			igl0 = 0;
		}		
	}


	//
	// G <> 0 term
	//

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int igl = igl0;igl< rho_basis->ngg;igl++)
	{
		gx_arr[igl] = sqrt(rho_basis->gg_uniq[igl] * tpiba2);
	}

	double *r_d = nullptr;
	double *rhoc_d = nullptr;
	double *rab_d = nullptr;
	double *aux_d = nullptr;
	double *drhocg_d = nullptr;

	if(this->device == base_device::GpuDevice) 
	{
		resmem_var_op()(r_d, mesh);
		resmem_var_op()(rhoc_d, mesh);
		resmem_var_op()(rab_d, mesh);

		resmem_var_op()(aux_d, mesh);
		resmem_var_op()(gx_arr_d, rho_basis->ngg);
		resmem_var_op()(drhocg_d, rho_basis->ngg);

		syncmem_var_h2d_op()(gx_arr_d, gx_arr.data(), rho_basis->ngg);
		syncmem_var_h2d_op()(r_d, r, mesh);
		syncmem_var_h2d_op()(rab_d, rab, mesh);
		syncmem_var_h2d_op()(rhoc_d, rhoc, mesh);
	}

	if(this->device == base_device::GpuDevice) 
	{
		hamilt::cal_stress_drhoc_aux_op<FPTYPE, Device>()(
				r_d,rhoc_d,gx_arr_d+igl0,rab_d,drhocg_d+igl0,mesh,igl0,rho_basis->ngg-igl0,omega,type);
		syncmem_var_d2h_op()(drhocg+igl0, drhocg_d+igl0, rho_basis->ngg-igl0);	

	} 
	else 
	{
		hamilt::cal_stress_drhoc_aux_op<FPTYPE, Device>()(
				r,rhoc,gx_arr.data()+igl0,rab,drhocg+igl0,mesh,igl0,rho_basis->ngg-igl0,omega,type);

	}
	delmem_var_op()(r_d);
    delmem_var_op()(rhoc_d);
    delmem_var_op()(rab_d);
    delmem_var_op()(gx_arr_d);
    delmem_var_op()(drhocg_d);

	return;
}

template class Stress_Func<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Stress_Func<double, base_device::DEVICE_GPU>;
#endif

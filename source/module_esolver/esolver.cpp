#include "esolver.h"
#include "module_base/module_device/device.h"
#include "esolver_ks_pw.h"
#include "esolver_sdft_pw.h"
#ifdef __LCAO
#include "esolver_ks_lcao.h"
#include "esolver_ks_lcao_tddft.h"
#endif
#include "esolver_dp.h"
#include "esolver_lj.h"
#include "esolver_of.h"
#include "module_md/md_para.h"

namespace ModuleESolver
{

void ESolver::printname(void)
{
    std::cout << classname << std::endl;
}

std::string determine_type(void)
{
	std::string esolver_type = "none";
	if (GlobalV::BASIS_TYPE == "pw")
	{
		if(GlobalV::ESOLVER_TYPE == "sdft")
		{
			esolver_type = "sdft_pw";
		}
		else if(GlobalV::ESOLVER_TYPE == "ofdft")
		{
			esolver_type = "ofdft";
		}
		else if(GlobalV::ESOLVER_TYPE == "ksdft")
		{
			esolver_type = "ksdft_pw";
		}
	}
	else if (GlobalV::BASIS_TYPE == "lcao_in_pw")
	{
#ifdef __LCAO
		if(GlobalV::ESOLVER_TYPE == "sdft")
		{
			esolver_type = "sdft_pw";
		}
		else if(GlobalV::ESOLVER_TYPE == "ksdft")
		{
			esolver_type = "ksdft_pw";
		}
#else
		ModuleBase::WARNING_QUIT("ESolver", "LCAO basis type must be compiled with __LCAO");
#endif
	}
	else if (GlobalV::BASIS_TYPE == "lcao")
	{
#ifdef __LCAO
		if(GlobalV::ESOLVER_TYPE == "tddft")
		{
			esolver_type = "ksdft_lcao_tddft";
		}
		else if(GlobalV::ESOLVER_TYPE == "ksdft")
		{
			esolver_type = "ksdft_lcao";
		}
#else
		ModuleBase::WARNING_QUIT("ESolver", "LCAO basis type must be compiled with __LCAO");
#endif
	}

	if(GlobalV::ESOLVER_TYPE == "lj")
	{
		esolver_type = "lj_pot";
	}
	else if(GlobalV::ESOLVER_TYPE == "dp")
	{
		esolver_type = "dp_pot";
	}
	else if(esolver_type == "none")
	{
		ModuleBase::WARNING_QUIT("ESolver", "No such esolver_type combined with basis_type");
	}

	GlobalV::ofs_running << " The esolver type has been set to : " << esolver_type << std::endl;

	auto device_info = GlobalV::device_flag;

	for (char &c : device_info) 
	{
		if (std::islower(c)) 
		{
			c = std::toupper(c);
		}
	}
	if (GlobalV::MY_RANK == 0) 
	{
		std::cout << " RUNNING WITH DEVICE  : " << device_info << " / "
			<< base_device::information::get_device_info(GlobalV::device_flag) << std::endl;
	}

	GlobalV::ofs_running << "\n RUNNING WITH DEVICE  : " << device_info << " / "
		<< base_device::information::get_device_info(GlobalV::device_flag) << std::endl;

	return esolver_type;
}


//Some API to operate E_Solver
void init_esolver(ESolver*& p_esolver)
{
	//determine type of esolver based on INPUT information
	std::string esolver_type = determine_type();

	//initialize the corresponding Esolver child class
	if (esolver_type == "ksdft_pw")
	{
#if ((defined __CUDA) || (defined __ROCM))
		if (GlobalV::device_flag == "gpu") 
		{
			if (GlobalV::precision_flag == "single") 
			{
                p_esolver = new ESolver_KS_PW<std::complex<float>, base_device::DEVICE_GPU>();
            }
			else 
			{
                p_esolver = new ESolver_KS_PW<std::complex<double>, base_device::DEVICE_GPU>();
            }
			return;
		}
#endif
		if (GlobalV::precision_flag == "single") 
		{
            p_esolver = new ESolver_KS_PW<std::complex<float>, base_device::DEVICE_CPU>();
        }
        else 
		{
            p_esolver = new ESolver_KS_PW<std::complex<double>, base_device::DEVICE_CPU>();
        }
    }
#ifdef __LCAO
	else if (esolver_type == "ksdft_lcao")
	{
		if (GlobalV::GAMMA_ONLY_LOCAL)
		{
			p_esolver = new ESolver_KS_LCAO<double, double>();
		}
		else if (GlobalV::NSPIN < 4)
		{
			p_esolver = new ESolver_KS_LCAO<std::complex<double>, double>();
		}
		else
		{
			p_esolver = new ESolver_KS_LCAO<std::complex<double>, std::complex<double>>();
		}
	}
	else if (esolver_type == "ksdft_lcao_tddft")
	{
		p_esolver = new ESolver_KS_LCAO_TDDFT();
	}
#endif
	else if (esolver_type == "sdft_pw")
	{
		p_esolver = new ESolver_SDFT_PW();
	}
	else if(esolver_type == "ofdft")
	{
		p_esolver = new ESolver_OF();
	}
	else if (esolver_type == "lj_pot")
	{
		p_esolver = new ESolver_LJ();
	}
	else if (esolver_type == "dp_pot")
	{
		p_esolver = new ESolver_DP(INPUT.mdp.pot_file);
	}
}


void clean_esolver(ESolver*& pesolver)
{
	delete pesolver;
}

}

//QO 2022-1-7
//This file contains subroutines for calculating O_delta, i.e., corrections of the bandgap,
#include "module_parameter/parameter.h"
//which is defind as sum_mu,nu rho^{hl}_mu,nu <chi_mu|alpha>V(D)<alpha|chi_nu>
//where rho^{hl}_mu,nu = C_{L\mu}C_{L\nu} - C_{H\mu}C_{H\nu}, L for LUMO, H for HOMO

//There are two subroutines in this file:
//1. cal_o_delta, which is used for gamma point calculation
//2. cal_o_delta_k, which is used for multi-k calculation

#ifdef __DEEPKS

#include "LCAO_deepks.h"
#include "module_base/parallel_reduce.h"

void LCAO_Deepks::cal_o_delta(const std::vector<std::vector<ModuleBase::matrix>>& dm_hl)
{
    ModuleBase::TITLE("LCAO_Deepks", "cal_o_delta");
    this->o_delta.zero_out();
    for (int hl = 0; hl < 1; ++hl)
    {
        for (int i = 0; i < PARAM.globalv.nlocal; ++i)
        {
            for (int j = 0; j < PARAM.globalv.nlocal; ++j)
            {
                const int mu = pv->global2local_row(j);
                const int nu = pv->global2local_col(i);
            
                if (mu >= 0 && nu >= 0)
                {                
                    const int index = nu * pv->nrow + mu;
                    for (int is = 0; is < PARAM.inp.nspin; ++is)
                    {
                        this->o_delta(0,hl) += dm_hl[hl][is](nu, mu) * this->H_V_delta[index];
                    }
                }
            }
        }
        Parallel_Reduce::reduce_all(this->o_delta(0, hl));
    }
    return;
}


//calculating the correction of (LUMO-HOMO) energies, i.e., band gap corrections
//for multi_k calculations
void LCAO_Deepks::cal_o_delta_k(const std::vector<std::vector<ModuleBase::ComplexMatrix>>& dm_hl,
    const int nks)
{
    ModuleBase::TITLE("LCAO_Deepks", "cal_o_delta_k");
    
    for(int ik=0; ik<nks; ik++)
    {
        for (int hl=0; hl<1; hl++)
        {
            std::complex<double> o_delta_k=std::complex<double>(0.0,0.0);
            for (int i = 0; i < PARAM.globalv.nlocal; ++i)
            {
                for (int j = 0; j < PARAM.globalv.nlocal; ++j)
                {
                    const int mu = pv->global2local_row(j);
                    const int nu = pv->global2local_col(i);
            
                    if (mu >= 0 && nu >= 0)
                    {                
                        int iic;
                        if(PARAM.inp.ks_solver=="genelpa" || PARAM.inp.ks_solver=="scalapack_gvx" || PARAM.inp.ks_solver=="pexsi")  // save the matrix as column major format
                        {
                            iic = mu + nu * pv->nrow;
                        }
                        else
                        {
                            iic = mu * pv->ncol + nu;
                        }
                        o_delta_k += dm_hl[hl][ik](nu, mu) * this->H_V_delta_k[ik][iic];
                    }
                } //end j
            } //end i
            Parallel_Reduce::reduce_all(o_delta_k);
            this->o_delta(ik,hl) = o_delta_k.real();
        }// end hl
    }// end nks
    
    return;
}

#endif

#include "FORCE_gamma.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_psi/psi.h"
#include "module_elecstate/cal_dm.h"
#include "module_elecstate/module_dm/cal_dm_psi.h"

// force due to the overlap matrix.
// need energy density matrix here.
void Force_LCAO_gamma::cal_foverlap(
	const bool isforce, 
    const bool isstress,
    const psi::Psi<double> *psid,
    const elecstate::ElecState *pelec,
    LCAO_Matrix &lm,
    ModuleBase::matrix &foverlap,
	ModuleBase::matrix &soverlap)
{
    ModuleBase::TITLE("Force_LCAO_gamma","cal_foverlap");
    ModuleBase::timer::tick("Force_LCAO_gamma","cal_foverlap");

    ModuleBase::timer::tick("Force_LCAO_gamma","cal_edm_2d");

    ModuleBase::matrix wgEkb;
    wgEkb.create(GlobalV::NSPIN, GlobalV::NBANDS);

    for(int is=0; is<GlobalV::NSPIN; is++)
    {
        for(int ib=0; ib<GlobalV::NBANDS; ib++)
        {
            wgEkb(is,ib) = pelec->wg(is,ib) * pelec->ekb(is, ib);
        }
    }

    // construct a DensityMatrix for Gamma-Only
    const Parallel_Orbitals* pv = this->ParaV;
    elecstate::DensityMatrix<double, double> EDM(pv,GlobalV::NSPIN);

    elecstate::cal_dm_psi(EDM.get_paraV_pointer(), wgEkb, psid[0], EDM);

    ModuleBase::timer::tick("Force_LCAO_gamma","cal_edm_2d");

    for(int i=0; i<GlobalV::NLOCAL; i++)
    {
        const int iat = GlobalC::ucell.iwt2iat[i];
        for(int j=0; j<GlobalV::NLOCAL; j++)
        {
            const int mu = pv->global2local_row(j);
            const int nu = pv->global2local_col(i);

            if(mu>=0 && nu>=0)
            {
                const int index = mu * pv->ncol + nu;
                double sum = 0.0;
                for(int is=0; is<GlobalV::NSPIN; ++is)
                {
                    //sum += edm_gamma[is](nu, mu);
                    sum += EDM.get_DMK(is+1, 0, nu, mu);
                }
                sum *= 2.0;

                if(isforce)
                {
                    foverlap(iat,0) += sum * lm.DSloc_x[index];
                    foverlap(iat,1) += sum * lm.DSloc_y[index];
                    foverlap(iat,2) += sum * lm.DSloc_z[index];
                }

                if(isstress)
                {
                    soverlap(0,0) += sum/2.0 * lm.DSloc_11[index];
                    soverlap(0,1) += sum/2.0 * lm.DSloc_12[index];
                    soverlap(0,2) += sum/2.0 * lm.DSloc_13[index];
                    soverlap(1,1) += sum/2.0 * lm.DSloc_22[index];
                    soverlap(1,2) += sum/2.0 * lm.DSloc_23[index];
                    soverlap(2,2) += sum/2.0 * lm.DSloc_33[index];
                }
            }

        }
    }

    if(isstress)
    {
		StressTools::stress_fill(GlobalC::ucell.lat0, GlobalC::ucell.omega, soverlap);
    }
    ModuleBase::timer::tick("Force_LCAO_gamma","cal_foverlap");
    return;
}

#include "esolver_of.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "module_parameter/parameter.h"

namespace ModuleESolver
{

/**
 * @brief [Interface to opt]
 * Initialize the opts
 */
void ESolver_OF::init_opt()
{
    if (this->opt_dcsrch_ == nullptr)
    {
        this->opt_dcsrch_ = new ModuleBase::Opt_DCsrch();
    }

    if (this->of_method_ == "tn")
    {
        if (this->opt_tn_ == nullptr)
        {
            this->opt_tn_ = new ModuleBase::Opt_TN();
        }
        this->opt_tn_->allocate(this->pw_rho->nrxx);
        this->opt_tn_->set_para(this->dV_);
    }
    else if (this->of_method_ == "cg1" || this->of_method_ == "cg2")
    {
        if (this->opt_cg_ == nullptr)
        {
            this->opt_cg_ = new ModuleBase::Opt_CG();
        }
        this->opt_cg_->allocate(this->pw_rho->nrxx);
        this->opt_cg_->set_para(this->dV_);
        this->opt_dcsrch_->set_paras(1e-4, 1e-2);
    }
    else if (this->of_method_ == "bfgs")
    {
        ModuleBase::WARNING_QUIT("esolver_of", "BFGS is not supported now.");
        return;
    }

    // optimize theta if nspin=2
    if (PARAM.inp.nspin == 2)
    {
        this->opt_cg_mag_ = new ModuleBase::Opt_CG;
        this->opt_cg_mag_->allocate(PARAM.inp.nspin);
    }
}

void ESolver_OF::cal_potential_wrapper(double* ptemp_phi, double* rdLdphi)
{
    this->bound_cal_potential_(ptemp_phi, rdLdphi);
}

/**
 * @brief [Interface to opt]
 * Call optimization methods to get the optimization direction
 */
void ESolver_OF::get_direction(UnitCell& ucell)
{
    for (int is = 0; is < PARAM.inp.nspin; ++is)
    {
        if (this->of_method_ == "tn")
        {
            this->tn_spin_flag_ = is;
            opt_tn_->next_direct(this->pphi_[is],
                                 this->pdLdphi_[is],
                                 this->flag_,
                                 this->pdirect_[is],
                                 this,
                                 &ESolver_OF::cal_potential_wrapper);
        }
        else if (this->of_method_ == "cg1")
        {
            opt_cg_->next_direct(this->pdLdphi_[is], 1, this->pdirect_[is]);
        }
        else if (this->of_method_ == "cg2")
        {
            opt_cg_->next_direct(this->pdLdphi_[is], 2, this->pdirect_[is]);
        }
        else if (this->of_method_ == "bfgs")
        {
            return;
        }
        else
        {
            ModuleBase::WARNING_QUIT("ESolver_OF", "of_method must be one of CG, TN, or BFGS.");
        }
    }
}

/**
 * @brief [Interface to opt]
 * Call line search to find the best step length
 *
 * @param dEdtheta d E / d theta
 * @param ptemp_phi
 * @param ucell
 */
void ESolver_OF::get_step_length(double* dEdtheta, double** ptemp_phi, UnitCell& ucell)
{
    double temp_energy = 0.0;      // energy of temp_phi and temp_rho
    double kinetic_energy = 0.0;   // kinetic energy
    double pseudopot_energy = 0.0; // electron-ion interaction energy

    if (PARAM.inp.nspin == 1)
    {
        int numDC = 0; // iteration number of line search
        strcpy(this->task_, "START");
        while (true)
        {
            // update energy
            this->pelec->cal_energies(2);
            temp_energy = this->pelec->f_en.etot;
            kinetic_energy = this->kedf_manager_->get_energy(); // kinetic energy
            pseudopot_energy = this->inner_product(this->pelec->pot->get_fixed_v(),
                                                   this->ptemp_rho_->rho[0],
                                                   this->pw_rho->nrxx,
                                                   this->dV_);
            Parallel_Reduce::reduce_all(pseudopot_energy);
            temp_energy += kinetic_energy + pseudopot_energy;

            // line search to update theta[0]
            this->opt_dcsrch_->dcSrch(temp_energy, dEdtheta[0], this->theta_[0], this->task_);
            numDC++;

            // decide what to do next according to the output of line search
            if (strncmp(this->task_, "FG", 2) == 0) // continue line search
            {
                // update tempPhi and tempRho
                for (int i = 0; i < this->pw_rho->nrxx; ++i)
                {
                    ptemp_phi[0][i]
                        = this->pphi_[0][i] * cos(this->theta_[0]) + this->pdirect_[0][i] * sin(this->theta_[0]);
                    this->ptemp_rho_->rho[0][i] = ptemp_phi[0][i] * ptemp_phi[0][i];
                }

                // get dEdtheta of new tempPhi and tempRho
                this->cal_dEdtheta(ptemp_phi, this->ptemp_rho_, ucell, this->theta_, dEdtheta);

                if (numDC > this->max_dcsrch_)
                {
                    GlobalV::ofs_warning << "ESolver_OF linesearch: WARNING "
                                         << "excedd the max iter number." << std::endl;
                    break;
                }
            }
            else if (strncmp(this->task_, "CO", 2) == 0) // convergence achieved
            {
                break;
            }
            else if (strncmp(this->task_, "WA", 2) == 0) // warning of line search
            {
                GlobalV::ofs_warning << "ESolver_OF linesearch: WARNING " << this->task_ << std::endl;
                std::cout << this->task_ << std::endl;
                break;
            }
            else if (strncmp(this->task_, "ER", 2) == 0) // ERROR in line search
            {
                GlobalV::ofs_warning << "ESolver_OF linesearch: ERROR " << this->task_ << std::endl;
                std::cout << this->task_ << std::endl;
                break;
            }
        }
    }
    else if (PARAM.inp.nspin == 2)
    {
        ModuleBase::WARNING_QUIT("esolver_of", "Sorry, SPIN2 case is not supported by OFDFT for now.");
        // ========================== Under testing ==========================
        //     this->opt_cg_mag_->refresh();

        //     double *pthetaDir = new double[PARAM.inp.nspin];
        //     double *temp_theta = new double[PARAM.inp.nspin];
        //     ModuleBase::GlobalFunc::ZEROS(pthetaDir, PARAM.inp.nspin);
        //     ModuleBase::GlobalFunc::ZEROS(temp_theta, PARAM.inp.nspin);
        //     double thetaAlpha = 0.;
        //     double alphaTol = 1e-4;
        //     double maxThetaDir = 0.;
        //     double dEdalpha = 0.;
        //     int thetaIter = 0;
        //     int numDC = 0;

        //     while (true)
        //     {
        //         this->opt_cg_mag_->next_direct(dEdtheta, 1, pthetaDir);

        //         dEdalpha = this->inner_product(dEdtheta, pthetaDir, 2, 1.);

        //         if (dEdalpha >= 0.)
        //         {
        //             for (int is = 0; is < PARAM.inp.nspin; ++is)
        //             {
        //                 pthetaDir[is] = -dEdtheta[is];
        //             }
        //             dEdalpha = this->inner_product(dEdtheta, pthetaDir, 2,
        //             1);
        //         }

        //         maxThetaDir = max(abs(pthetaDir[0]), abs(pthetaDir[1]));
        //         thetaAlpha = min(0.1, 0.1*ModuleBase::PI/maxThetaDir);

        //         // line search along thetaDir to find thetaAlpha
        //         this->opt_dcsrch_->set_paras(1e-4, 1e-2, 1e-12, 0.,
        //         ModuleBase::PI/maxThetaDir); strcpy(this->task_, "START");
        //         numDC = 0;
        //         while(true)
        //         {
        //             this->pelec->f_en.calculate_etot(this->pw_rho->nrxx,
        //             this->pw_rho->nxyz); temp_energy =
        //             this->pelec->f_en.etot; kinetic_energy =
        //             this->kinetic_energy(); pseudopot_energy = 0.; for (int
        //             is = 0; is < PARAM.inp.nspin; ++is) {
        //                 pseudopot_energy +=
        //                 this->inner_product(GlobalC::pot.vltot,
        //                 ptemp_rho_[is], this->pw_rho->nrxx, this->dV_);
        //             }
        //             Parallel_Reduce::reduce_all(pseudopot_energy);
        //             temp_energy += kinetic_energy + pseudopot_energy;
        //             this->opt_dcsrch_->dcSrch(temp_energy, dEdalpha,
        //             thetaAlpha, this->task_); numDC++;

        //             if (strncmp(this->task_, "FG", 2) == 0)
        //             {
        //                 for (int is = 0; is < PARAM.inp.nspin; ++is)
        //                 {
        //                     temp_theta[is] = this->theta_[is] + thetaAlpha *
        //                     pthetaDir[is]; for (int ir = 0; ir <
        //                     this->pw_rho->nrxx; ++ir)
        //                     {
        //                         ptemp_phi[is][ir] = this->pphi_[is][ir] *
        //                         cos(temp_theta[is]) + this->pdirect_[is][ir]
        //                         * sin(temp_theta[is]); ptemp_rho_[is][ir] =
        //                         ptemp_phi[is][ir]
        //                         * ptemp_phi[is][ir];
        //                     }
        //                 }
        //                 this->cal_dEdtheta(ptemp_phi, ptemp_rho_, temp_theta,
        //                 dEdtheta); dEdalpha = this->inner_product(dEdtheta,
        //                 pthetaDir, 2, 1);

        //                 if (numDC > 10)
        //                 {
        //                     GlobalV::ofs_warning << "ESolver_OF linesearch:
        //                     WARNING " << "excedd the max iter number." <<
        //                     endl; break;
        //                 }
        //             }
        //             else if (strncmp(this->task_, "CO", 2) == 0)
        //             {
        //                 break;
        //             }
        //             else if (strncmp(this->task_, "WA", 2) == 0)
        //             {
        //                 GlobalV::ofs_warning << "ESolver_OF linesearch:
        //                 WARNING " << this->task_ << std::endl; cout <<
        //                 this->task_ << endl; break;
        //             }
        //             else if (strncmp(this->task_, "ER", 2) == 0)
        //             {
        //                 GlobalV::ofs_warning << "ESolver_OF linesearch: ERROR
        //                 " << this->task_ << std::endl; cout << this->task_ <<
        //                 endl; break;
        //             }
        //         }

        //         for (int is = 0; is < PARAM.inp.nspin; ++is) this->theta_[is]
        //         += thetaAlpha * pthetaDir[is]; if (sqrt(dEdtheta[0] *
        //         dEdtheta[0] + dEdtheta[1] * dEdtheta[1]) < alphaTol) break;
        //         thetaIter++;
        //         if (thetaIter > 2) break;
        //     }
        //     delete[] temp_theta;
        //     delete[] pthetaDir;
        // ========================== Under testing ==========================
    }
    else if (PARAM.inp.nspin == 4)
    {
        ModuleBase::WARNING_QUIT("esolver_of", "Sorry, SPIN4 case is not supported by OFDFT for now.");
    }
}
} // namespace ModuleESolver

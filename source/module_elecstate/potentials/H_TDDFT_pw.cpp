#include "module_base/timer.h"
#include "H_TDDFT_pw.h"
#include "src_lcao/ELEC_evolve.h"

namespace elecstate
{

int H_TDDFT_pw::istep = -1;
//==========================================================
// this function aims to add external time-dependent potential
// (eg: linear potential) used in tddft
// fuxiang add in 2017-05
//==========================================================
void H_TDDFT_pw::cal_fixed_v(double* vl_pseudo)
{
    ModuleBase::TITLE("H_TDDFT_pw", "cal_fixed_v");

    //time evolve
    H_TDDFT_pw::istep++;

    //judgement to skip vext
    if(ELEC_evolve::td_vext == 0 || double(H_TDDFT_pw::istep) >= ELEC_evolve::td_timescale)
    {
        return;
    }

    ModuleBase::timer::tick("H_TDDFT_pw", "cal_fixed_v");

    //====================================================
    // add external linear potential, fuxiang add in 2017/05
    //====================================================
    double* vextold = new double[this->rho_basis_->nrxx];
    double* vext = new double[this->rho_basis_->nrxx];
    const int yz = this->rho_basis_->ny * this->rho_basis_->nplane;
    int index, i, j, k;

    for (int ir = 0; ir < this->rho_basis_->nrxx; ++ir)
    {
        index = ir;
        i = index / yz; // get the z, z is the fastest
        index = index - yz * i; // get (x,y)
        j = index / this->rho_basis_->nplane; // get y
        k = index - this->rho_basis_->nplane * j + this->rho_basis_->startz_current; // get x

        if (ELEC_evolve::td_vext_dire == 1)
        {
            if (k < this->rho_basis_->nx * 0.05)
            {
                vextold[ir] = (0.019447 * k / this->rho_basis_->nx - 0.001069585) * this->ucell_->lat0;
            }
            else if (k >= this->rho_basis_->nx * 0.05 && k < this->rho_basis_->nx * 0.95)
            {
                vextold[ir] = -0.0019447 * k / this->rho_basis_->nx * this->ucell_->lat0;
            }
            else if (k >= this->rho_basis_->nx * 0.95)
            {
                vextold[ir] = (0.019447 * (1.0 * k / this->rho_basis_->nx - 1) - 0.001069585) * this->ucell_->lat0;
            }
        }
        else if (ELEC_evolve::td_vext_dire == 2)
        {
            if (j < this->rho_basis_->nx * 0.05)
            {
                vextold[ir] = (0.019447 * j / this->rho_basis_->nx - 0.001069585) * this->ucell_->lat0;
            }
            else if (j >= this->rho_basis_->nx * 0.05 && j < this->rho_basis_->nx * 0.95)
            {
                vextold[ir] = -0.0019447 * j / this->rho_basis_->nx * this->ucell_->lat0;
            }
            else if (j >= this->rho_basis_->nx * 0.95)
            {
                vextold[ir] = (0.019447 * (1.0 * j / this->rho_basis_->nx - 1) - 0.001069585) * this->ucell_->lat0;
            }
        }
        else if (ELEC_evolve::td_vext_dire == 3)
        {
            if (i < this->rho_basis_->nx * 0.05)
            {
                vextold[ir] = (0.019447 * i / this->rho_basis_->nx - 0.001069585) * this->ucell_->lat0;
            }
            else if (i >= this->rho_basis_->nx * 0.05 && i < this->rho_basis_->nx * 0.95)
            {
                vextold[ir] = -0.0019447 * i / this->rho_basis_->nx * this->ucell_->lat0;
            }
            else if (i >= this->rho_basis_->nx * 0.95)
            {
                vextold[ir] = (0.019447 * (1.0 * i / this->rho_basis_->nx - 1) - 0.001069585) * this->ucell_->lat0;
            }
        }

        // Gauss
        if (ELEC_evolve::td_vexttype == 1)
        {
            const double w = 22.13; // eV
            const double sigmasquare = 700;
            const double timecenter = 700;
            // Notice: these three parameters should be written in INPUT. I will correct soon.
            const double timenow
                = (H_TDDFT_pw::istep - timecenter) * ELEC_evolve::td_scf_thr * 41.34; // 41.34 is conversion factor of fs-a.u.
            vext[ir] = vextold[ir] * cos(w / 27.2116 * timenow) * exp(-timenow * timenow * 0.5 / (sigmasquare))
                        * 0.25; // 0.1 is modified in 2018/1/12
        }

        // HHG of H atom
        if (ELEC_evolve::td_vexttype == 2)
        {
            const double w_h = 0.0588; // a.u.
            const int stepcut1 = 1875;
            const int stepcut2 = 5625;
            const int stepcut3 = 7500;
            // The parameters should be written in INPUT!
            if (H_TDDFT_pw::istep < stepcut1)
            {
                vext[ir] = vextold[ir] * 2.74 * H_TDDFT_pw::istep / stepcut1
                            * cos(w_h * H_TDDFT_pw::istep * ELEC_evolve::td_scf_thr * 41.34); // 2.74 is equal to E0;
            }
            else if (H_TDDFT_pw::istep < stepcut2)
            {
                vext[ir] = vextold[ir] * 2.74 * cos(w_h * H_TDDFT_pw::istep * ELEC_evolve::td_scf_thr * 41.34);
            }
            else if (H_TDDFT_pw::istep < stepcut3)
            {
                vext[ir] = vextold[ir] * 2.74 * (stepcut3 - H_TDDFT_pw::istep) / stepcut1
                            * cos(w_h * H_TDDFT_pw::istep * ELEC_evolve::td_scf_thr * 41.34);
            }
        }

        // HHG of H2
        //  Type 3 will be modified into more normolized type soon.
        if (ELEC_evolve::td_vexttype == 3)
        {
            const double w_h2 = 0.0428; // a.u.
            const double w_h3 = 0.00107; // a.u.
            const double timenow = (H_TDDFT_pw::istep)*ELEC_evolve::td_scf_thr * 41.34;
            // The parameters should be written in INPUT!

            // vext[ir] = vextold[ir]*2.74*cos(0.856*timenow)*sin(0.0214*timenow)*sin(0.0214*timenow);
            // vext[ir] = vextold[ir]*2.74*cos(0.856*timenow)*sin(0.0214*timenow)*sin(0.0214*timenow)*0.01944;
            vext[ir] = vextold[ir] * 2.74 * cos(w_h2 * timenow) * sin(w_h3 * timenow) * sin(w_h3 * timenow);
        }

        vl_pseudo[ir] += vext[ir];

        // std::cout << "x: " << k <<"	" << "y: " << j <<"	"<< "z: "<< i <<"	"<< "ir: " << ir << std::endl;
        // std::cout << "vext: " << vext[ir] << std::endl;
        // std::cout << "vrs: " << vrs(is,ir) <<std::endl;
    }
    std::cout << "For time step "<<H_TDDFT_pw::istep<<" :"<<" vext exists" << std::endl;

    delete[] vextold;
    delete[] vext;

    ModuleBase::timer::tick("H_TDDFT_pw", "cal_fixed_v");
    return;
} // end subroutine set_vrs_tddft

} // namespace elecstate

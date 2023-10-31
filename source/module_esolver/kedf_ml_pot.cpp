#include "./kedf_ml.h"
#include "../src_parallel/parallel_reduce.h"
#include "../module_base/global_function.h"

double KEDF_ML::potGammaTerm(int ir)
{
    return (GlobalV::of_ml_gamma) ? 1./3. * gamma[ir] * this->temp_gradient[ir * this->ninput + this->nn_input_index["gamma"]] : 0.;
}

double KEDF_ML::potPTerm1(int ir)
{
    return (GlobalV::of_ml_p) ? - 8./3. * p[ir] * this->temp_gradient[ir * this->ninput + this->nn_input_index["p"]] : 0.;
}

double KEDF_ML::potQTerm1(int ir)
{
    return (GlobalV::of_ml_q) ? - 5./3. * q[ir] * this->temp_gradient[ir * this->ninput + this->nn_input_index["q"]] : 0.;
}

double KEDF_ML::potXiTerm1(int ir)
{
    return (GlobalV::of_ml_xi) ? -1./3. * xi[ir] * this->temp_gradient[ir * this->ninput + this->nn_input_index["xi"]] : 0.;
}

double KEDF_ML::potTanhxiTerm1(int ir)
{
    return (GlobalV::of_ml_tanhxi) ? -1./3. * xi[ir] * this->ml_data->dtanh(this->tanhxi[ir], this->chi_xi)
                                    * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhxi"]] : 0.;
}

double KEDF_ML::potTanhpTerm1(int ir)
{
    return (GlobalV::of_ml_tanhp) ? - 8./3. * p[ir] * this->ml_data->dtanh(this->tanhp[ir], this->chi_p)
                                 * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhp"]] : 0.;
}

double KEDF_ML::potTanhqTerm1(int ir)
{
    return (GlobalV::of_ml_tanhq) ? - 5./3. * q[ir] * this->ml_data->dtanh(this->tanhq[ir], this->chi_q)
                                 * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhq"]] : 0.;
}

void KEDF_ML::potGammanlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rGammanlTerm)
{
    if (!GlobalV::of_ml_gammanl) return;
    double *dFdgammanl = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdgammanl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->temp_gradient[ir * this->ninput + this->nn_input_index["gammanl"]];
    }
    this->ml_data->multiKernel(dFdgammanl, pw_rho, rGammanlTerm.data());
    for (int ir = 0; ir < this->nx; ++ir)
    {
        rGammanlTerm[ir] *= 1./3. * this->gamma[ir] / prho[0][ir];
    }
    delete[] dFdgammanl;
}

void KEDF_ML::potXinlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rXinlTerm)
{
    if (!GlobalV::of_ml_xi) return;
    double *dFdxi = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdxi[ir] = this->cTF * pow(prho[0][ir], 4./3.) * this->temp_gradient[ir * this->ninput + this->nn_input_index["xi"]];
    }
    this->ml_data->multiKernel(dFdxi, pw_rho, rXinlTerm.data());
    for (int ir = 0; ir < this->nx; ++ir)
    {
        rXinlTerm[ir] *= 1./3. * pow(prho[0][ir], -2./3.);
    }
    delete[] dFdxi;
}

void KEDF_ML::potTanhxinlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhxinlTerm)
{
    if (!GlobalV::of_ml_tanhxi) return;
    double *dFdxi = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdxi[ir] = this->cTF * pow(prho[0][ir], 4./3.) * this->ml_data->dtanh(this->tanhxi[ir], this->chi_xi)
                    * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhxi"]];
    }
    this->ml_data->multiKernel(dFdxi, pw_rho, rTanhxinlTerm.data());
    for (int ir = 0; ir < this->nx; ++ir)
    {
        rTanhxinlTerm[ir] *= 1./3. * pow(prho[0][ir], -2./3.);
    }
    delete[] dFdxi;
}

void KEDF_ML::potTanhxi_nlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhxi_nlTerm)
{
    if (!GlobalV::of_ml_tanhxi_nl) return;
    double *dFdxi = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdxi[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhxi_nl"]];
    }
    this->ml_data->multiKernel(dFdxi, pw_rho, dFdxi);
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdxi[ir] *= this->ml_data->dtanh(this->tanhxi[ir], this->chi_xi) / pow(prho[0][ir], 1./3.);
    }
    this->ml_data->multiKernel(dFdxi, pw_rho, rTanhxi_nlTerm.data());
    for (int ir = 0; ir < this->nx; ++ir)
    {
        rTanhxi_nlTerm[ir] += - dFdxi[ir] * this->xi[ir];
        rTanhxi_nlTerm[ir] *= 1./3. * pow(prho[0][ir], -2./3.);
    }
    delete[] dFdxi;
}

// get contribution of p and pnl
void KEDF_ML::potPPnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rPPnlTerm)
{
    if (!GlobalV::of_ml_p && !GlobalV::of_ml_pnl) return;
    // cout << "begin p" << endl;
    double *dFdpnl = nullptr;
    if (GlobalV::of_ml_pnl)
    {
        dFdpnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdpnl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->temp_gradient[ir * this->ninput + this->nn_input_index["pnl"]];
        }
        this->ml_data->multiKernel(dFdpnl, pw_rho, dFdpnl);
    }

    double ** tempP = new double*[3];
    for (int i = 0; i < 3; ++i)
    {
        tempP[i] = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            tempP[i][ir] = (GlobalV::of_ml_p)? - 3./20. * this->temp_gradient[ir * this->ninput + this->nn_input_index["p"]] * nablaRho[i][ir] / prho[0][ir] * /*Ha2Ry*/ 2. : 0.;
            if (GlobalV::of_ml_pnl)
            {
                tempP[i][ir] += - this->pqcoef * 2. * this->nablaRho[i][ir] / pow(prho[0][ir], 8./3.) * dFdpnl[ir];
            }
        }
    }
    this->ml_data->divergence(tempP, pw_rho, rPPnlTerm.data());

    if (GlobalV::of_ml_pnl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rPPnlTerm[ir] += -8./3. * this->p[ir]/prho[0][ir] * dFdpnl[ir];
        }
    }

    for (int i = 0; i < 3; ++i) delete[] tempP[i];
    delete[] tempP;
    delete[] dFdpnl;
}

void KEDF_ML::potQQnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rQQnlTerm)
{
    if (!GlobalV::of_ml_q && !GlobalV::of_ml_qnl) return;
    double *dFdqnl = nullptr;
    if (GlobalV::of_ml_qnl)
    {
        dFdqnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdqnl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->temp_gradient[ir * this->ninput + this->nn_input_index["qnl"]];
        }
        this->ml_data->multiKernel(dFdqnl, pw_rho, dFdqnl);
    }

    double * tempQ = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        tempQ[ir] = (GlobalV::of_ml_q)? 3./40. * this->temp_gradient[ir * this->ninput + this->nn_input_index["q"]] * /*Ha2Ry*/ 2. : 0.;
        if (GlobalV::of_ml_qnl)
        {
            tempQ[ir] += this->pqcoef / pow(prho[0][ir], 5./3.) * dFdqnl[ir];
        }
    }
    this->ml_data->Laplacian(tempQ, pw_rho, rQQnlTerm.data());

    if (GlobalV::of_ml_qnl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rQQnlTerm[ir] += -5./3. * this->q[ir] / prho[0][ir] * dFdqnl[ir];
        }
    }
    delete[] tempQ;
    delete[] dFdqnl;
}

void KEDF_ML::potTanhpTanh_pnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhpTanh_pnlTerm)
{
    if (!GlobalV::of_ml_tanhp && !GlobalV::of_ml_tanh_pnl) return;
    // Note we assume that tanhp_nl and tanh_pnl will NOT be used together.
    if (GlobalV::of_ml_tanhp_nl) return;
    // cout << "begin tanhp" << endl;
    double *dFdpnl = nullptr;
    if (GlobalV::of_ml_tanh_pnl)
    {
        dFdpnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdpnl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->ml_data->dtanh(this->tanh_pnl[ir], this->chi_pnl)
                         * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanh_pnl"]];
        }
        this->ml_data->multiKernel(dFdpnl, pw_rho, dFdpnl);
    }

    double ** tempP = new double*[3];
    for (int i = 0; i < 3; ++i)
    {
        tempP[i] = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            tempP[i][ir] = (GlobalV::of_ml_tanhp)? - 3./20. * this->ml_data->dtanh(this->tanhp[ir], this->chi_p)
                           * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhp"]] * nablaRho[i][ir] / prho[0][ir] * /*Ha2Ry*/ 2. : 0.;
            if (GlobalV::of_ml_tanh_pnl)
            {
                tempP[i][ir] += - this->pqcoef * 2. * this->nablaRho[i][ir] / pow(prho[0][ir], 8./3.) * dFdpnl[ir];
            }
        }
    }
    this->ml_data->divergence(tempP, pw_rho, rTanhpTanh_pnlTerm.data());

    if (GlobalV::of_ml_tanh_pnl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rTanhpTanh_pnlTerm[ir] += -8./3. * this->p[ir]/prho[0][ir] * dFdpnl[ir];
        }
    }

    for (int i = 0; i < 3; ++i) delete[] tempP[i];
    delete[] tempP;
    delete[] dFdpnl;
}

void KEDF_ML::potTanhqTanh_qnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhqTanh_qnlTerm)
{
    if (!GlobalV::of_ml_tanhq && !GlobalV::of_ml_tanh_qnl) return;
    // Note we assume that tanhq_nl and tanh_qnl will NOT be used together.
    if (GlobalV::of_ml_tanhq_nl) return;
    double *dFdqnl = nullptr;
    if (GlobalV::of_ml_tanh_qnl)
    {
        dFdqnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdqnl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->ml_data->dtanh(this->tanh_qnl[ir], this->chi_qnl)
                         * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanh_qnl"]];
        }
        this->ml_data->multiKernel(dFdqnl, pw_rho, dFdqnl);
    }

    double * tempQ = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        tempQ[ir] = (GlobalV::of_ml_tanhq)? 3./40. * this->ml_data->dtanh(this->tanhq[ir], this->chi_q)
                    * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhq"]] * /*Ha2Ry*/ 2. : 0.;
        if (GlobalV::of_ml_tanh_qnl)
        {
            tempQ[ir] += this->pqcoef / pow(prho[0][ir], 5./3.) * dFdqnl[ir];
        }
    }
    this->ml_data->Laplacian(tempQ, pw_rho, rTanhqTanh_qnlTerm.data());

    if (GlobalV::of_ml_tanh_qnl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rTanhqTanh_qnlTerm[ir] += -5./3. * this->q[ir] / prho[0][ir] * dFdqnl[ir];
        }
    }
    delete[] tempQ;
    delete[] dFdqnl;
}

// Note we assume that tanhp_nl and tanh_pnl will NOT be used together.
void KEDF_ML::potTanhpTanhp_nlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhpTanhp_nlTerm)
{
    if (!GlobalV::of_ml_tanhp_nl) return;
    // cout << "begin tanhp" << endl;
    double *dFdpnl = nullptr;
    if (GlobalV::of_ml_tanhp_nl)
    {
        dFdpnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdpnl[ir] = this->cTF * pow(prho[0][ir], 5./3.)
                         * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhp_nl"]];
        }
        this->ml_data->multiKernel(dFdpnl, pw_rho, dFdpnl);
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdpnl[ir] *= this->ml_data->dtanh(this->tanhp[ir], this->chi_p);
        }
    }

    double ** tempP = new double*[3];
    for (int i = 0; i < 3; ++i)
    {
        tempP[i] = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            tempP[i][ir] = (GlobalV::of_ml_tanhp)? - 3./20. * this->ml_data->dtanh(this->tanhp[ir], this->chi_p)
                           * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhp"]] * nablaRho[i][ir] / prho[0][ir] * /*Ha2Ry*/ 2. : 0.;
            if (GlobalV::of_ml_tanhp_nl)
            {
                tempP[i][ir] += - this->pqcoef * 2. * this->nablaRho[i][ir] / pow(prho[0][ir], 8./3.) * dFdpnl[ir];
            }
        }
    }
    this->ml_data->divergence(tempP, pw_rho, rTanhpTanhp_nlTerm.data());

    if (GlobalV::of_ml_tanhp_nl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rTanhpTanhp_nlTerm[ir] += -8./3. * this->p[ir]/prho[0][ir] * dFdpnl[ir];
        }
    }

    for (int i = 0; i < 3; ++i) delete[] tempP[i];
    delete[] tempP;
    delete[] dFdpnl;
}

void KEDF_ML::potTanhqTanhq_nlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhqTanhq_nlTerm)
{
    if (!GlobalV::of_ml_tanhq_nl) return;

    double *dFdqnl = nullptr;
    if (GlobalV::of_ml_tanhq_nl)
    {
        dFdqnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdqnl[ir] = this->cTF * pow(prho[0][ir], 5./3.)
                         * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhq_nl"]];
        }
        this->ml_data->multiKernel(dFdqnl, pw_rho, dFdqnl);
    }

    double * tempQ = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        tempQ[ir] = (GlobalV::of_ml_tanhq)? 3./40. * this->ml_data->dtanh(this->tanhq[ir], this->chi_q)
                    * this->temp_gradient[ir * this->ninput + this->nn_input_index["tanhq"]] * /*Ha2Ry*/ 2. : 0.;
        if (GlobalV::of_ml_tanhq_nl)
        {
            dFdqnl[ir] *= this->ml_data->dtanh(this->tanhq[ir], this->chi_q);
            tempQ[ir] += this->pqcoef / pow(prho[0][ir], 5./3.) * dFdqnl[ir];
        }
    }
    this->ml_data->Laplacian(tempQ, pw_rho, rTanhqTanhq_nlTerm.data());

    if (GlobalV::of_ml_tanhq_nl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rTanhqTanhq_nlTerm[ir] += -5./3. * this->q[ir] / prho[0][ir] * dFdqnl[ir];
        }
    }
    delete[] tempQ;
    delete[] dFdqnl;
}
#include "./kedf_tf.h"

#include "module_parameter/parameter.h"
#include <iostream>

#include "module_base/parallel_reduce.h"

void KEDF_TF::set_para(int nx, double dV, double tf_weight)
{
    this->nx_ = nx;
    this->dV_ = dV;
    this->tf_weight_ = tf_weight;
}

/**
 * @brief Get the energy of TF KEDF,
 * \f[ E_{TF} = c_{TF} * \int{\rho^{5/3} dr} \f]
 *
 * @param prho charge density
 * @return the energy of TF KEDF
 */
double KEDF_TF::get_energy(const double* const* prho)
{
    double energy = 0.; // in Ry
    if (PARAM.inp.nspin == 1)
    {
        for (int ir = 0; ir < this->nx_; ++ir)
        {
            energy += std::pow(prho[0][ir], 5. / 3.);
        }
        energy *= this->dV_ * this->c_tf_;
    }
    else if (PARAM.inp.nspin == 2)
    {
        for (int is = 0; is < PARAM.inp.nspin; ++is)
        {
            for (int ir = 0; ir < this->nx_; ++ir)
            {
                energy += std::pow(2. * prho[is][ir], 5. / 3.);
            }
        }
        energy *= 0.5 * this->dV_ * this->c_tf_ * this->tf_weight_;
    }
    this->tf_energy = energy;
    Parallel_Reduce::reduce_all(this->tf_energy);
    return energy;
}

/**
 * @brief Get the energy density of TF KEDF
 * \f[ \tau_{TF} = c_{TF} * \rho^{5/3} \f]
 *
 * @param prho charge density
 * @param is the index of spin
 * @param ir the index of real space grid
 * @return the energy density of TF KEDF
 */
double KEDF_TF::get_energy_density(const double* const* prho, int is, int ir)
{
    double energyDen = 0.; // in Ry
    energyDen = this->c_tf_ * std::pow(prho[is][ir], 5. / 3.) * this->tf_weight_;
    return energyDen;
}

/**
 * @brief Get the kinetic energy of TF KEDF, and add it onto rtau_tf
 * \f[ \tau_{TF} = c_{TF} * \prho^{5/3} \f]
 * 
 * @param prho charge density
 * @param rtau_tf rtau_tf => rtau_tf + tau_tf
 */
void KEDF_TF::tau_tf(const double* const* prho, double* rtau_tf)
{
    if (PARAM.inp.nspin == 1)
    {
        for (int ir = 0; ir < this->nx_; ++ir)
        {
            rtau_tf[ir] += this->c_tf_ * std::pow(prho[0][ir], 5.0 / 3.0);
        }
    }
    else if (PARAM.inp.nspin == 2)
    {
        for (int is = 0; is < PARAM.inp.nspin; ++is)
        {
            for (int ir = 0; ir < this->nx_; ++ir)
            {
                rtau_tf[ir] += 0.5 * this->c_tf_ * std::pow(2.0 * prho[is][ir], 5.0 / 3.0);
            }
        }
    }
}

/**
 * @brief Get the potential of TF KEDF, and add it into rpotential,
 * and the TF energy will be calculated and stored in this->tf_energy
 * \f[ V_{TF} = \delta E_{TF}/\delta \rho = 5/3 * c_{TF} * \rho^{2/3} \f]
 *
 * @param prho charge density
 * @param rpotential rpotential => rpotential + V_{TF}
 */
void KEDF_TF::tf_potential(const double* const* prho, ModuleBase::matrix& rpotential)
{
    ModuleBase::timer::tick("KEDF_TF", "tf_potential");
    if (PARAM.inp.nspin == 1)
    {
        for (int ir = 0; ir < this->nx_; ++ir)
        {
            rpotential(0, ir) += 5.0 / 3.0 * this->c_tf_ * std::pow(prho[0][ir], 2. / 3.) * this->tf_weight_;
        }
    }
    else if (PARAM.inp.nspin == 2)
    {
        for (int is = 0; is < PARAM.inp.nspin; ++is)
        {
            for (int ir = 0; ir < this->nx_; ++ir)
            {
                rpotential(is, ir) += 5.0 / 3.0 * this->c_tf_ * std::pow(2. * prho[is][ir], 2. / 3.) * this->tf_weight_;
            }
        }
    }

    this->get_energy(prho);

    ModuleBase::timer::tick("KEDF_TF", "tf_potential");
}

/**
 * @brief Get the stress of TF KEDF, and store it into this->stress
 *
 * @param cell_vol the volume of cell
 */
void KEDF_TF::get_stress(double cell_vol)
{
    double temp = 0.;
    temp = 2. * this->tf_energy / (3. * cell_vol);

    for (int i = 0; i < 3; ++i)
    {
        this->stress(i, i) = temp;
    }
}
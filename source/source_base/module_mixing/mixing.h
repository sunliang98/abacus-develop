#ifndef MIXING_H_
#define MIXING_H_
#include <functional>

#include "mixing_data.h"

namespace Base_Mixing
{

/**
 * @brief Mixing class can mixing different steps of data to solver the iteration problem.
 *        For equation x = f(x), we can use iterative process to solve it:
 *                x_{n+1} = f(x_n), n = 0, 1, 2, ...
 *        To acceralte the convergence, we can use mixing method, we need information including
 *        x_in, x_out=f(x_in) in each iterative step.
 *
 */
class Mixing
{
  public:
    Mixing(){};
    virtual ~Mixing(){};

    /**
     * @brief init mixing data
     *
     * @param mdata mixing data
     * @param length the length of each vector
     * @param type_size size of type
     *
     */
    virtual void init_mixing_data(Mixing_Data& mdata, const int& length, const size_t& type_size) const
    {
        mdata.resize(data_ndim, length, type_size);
    }

    /**
     * @brief
     *
     * @param mdata store information of this iterative step
     * @param data_in x_in
     * @param data_out x_out = f(x_in)
     * @param screen pointer to the screen function for Ker-Ker mixing
     * @param mix (double* out, const double* in, const double* residual) calculate 'out' with 'in' and residual
     * @param need_calcoef whether need to calculate the coef
     *
     */
    virtual void push_data(Mixing_Data& mdata,
                           const double* data_in,
                           const double* data_out,
                           std::function<void(double*)> screen,
                           std::function<void(double*, const double*, const double*)> mix,
                           const bool& need_calcoef)
        = 0;
    virtual void push_data(
        Mixing_Data& mdata,
        const std::complex<double>* data_in,
        const std::complex<double>* data_out,
        std::function<void(std::complex<double>*)> screen,
        std::function<void(std::complex<double>*, const std::complex<double>*, const std::complex<double>*)> mix,
        const bool& need_calcoef)
        = 0;

    /**
     * @brief
     *
     * @param mdata store information of this iterative step
     * @param data_in x_in
     * @param data_out x_out = f(x_in)
     * @param screen pointer to the screen function for Ker-Ker mixing
     * @param need_calcoef whether need to calculate the coef
     *
     */
    virtual void push_data(Mixing_Data& mdata,
                           const double* data_in,
                           const double* data_out,
                           std::function<void(double*)> screen,
                           const bool& need_calcoef);
    virtual void push_data(Mixing_Data& mdata,
                           const std::complex<double>* data_in,
                           const std::complex<double>* data_out,
                           std::function<void(std::complex<double>*)> screen,
                           const bool& need_calcoef);

    /**
     * @brief calculate coeficients for mixing
     *
     * @param mdata Mixing_Data
     * @param inner_product pointer to the inner dot function
     */
    virtual void cal_coef(const Mixing_Data& mdata, std::function<double(double*, double*)> inner_product) = 0;
    virtual void cal_coef(const Mixing_Data& mdata,
                          std::function<double(std::complex<double>*, std::complex<double>*)> inner_product)
        = 0;

    /**
     * @brief calculate the mixing data
     *
     * @param mdata Mixing_Data
     * @param data_mix output data
     */
    void mix_data(const Mixing_Data& mdata, double* data_mix);
    void mix_data(const Mixing_Data& mdata, std::complex<double>* data_mix);

    /**
     * @brief reset mixing
     *
     */
    virtual void reset() = 0;

  public:
    // mixing_beta from INPUT
    double mixing_beta = 0.7;
    // coeficients for mixing
    std::vector<double> coef;
    // ndim for mixing
    int data_ndim = 1;
};

} // namespace Base_Mixing

#endif
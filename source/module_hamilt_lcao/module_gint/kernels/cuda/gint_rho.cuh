#ifndef GINT_RHO_CUH
#define GINT_RHO_CUH

#include <cuda_runtime.h>
namespace GintKernel
{

/**
 * @brief CUDA kernel to calculate psir.
 *
 * This kernel calculates the wave function psir using the provided input
 * parameters.
 *
 * @param ylmcoef pointer to the array of Ylm coefficients.
 * @param delta_r_g value of delta_r_g.
 * @param bxyz_g number of meshcells in a bigcell.
 * @param nwmax_g maximum nw.
 * @param input_double `double` type datas used to calculate psir.
 * @param input_int `int` type datas used to calculate psir.
 * @param num_psir  number of atoms on each bigcell.
 * @param psi_size_max maximum number of atoms on bigcell.
 * @param ucell_atom_nwl nw of each type of atom.
 * @param atom_iw2_new
 * @param atom_iw2_ylm
 * @param atom_nw pointer to the array of atom_nw values.
 * @param nr_max
 * @param psi_u
 * @param psir_ylm
 */
__global__ void get_psi(const double* const ylmcoef,
                        double delta_r_g,
                        int bxyz_g,
                        double nwmax_g,
                        const double* const input_double,
                        const int* const input_int,
                        const int* const num_psir,
                        int psi_size_max,
                        const int* const ucell_atom_nwl,
                        const bool* const atom_iw2_new,
                        const int* const atom_iw2_ylm,
                        const int* const atom_nw,
                        int nr_max,
                        const double* const psi_u,
                        double* psir_ylm);

/**
 * @brief Kernel function to calculate batch vector dot products.
 *
 * @param n             vector length.
 * @param vec_l_g       pointers to left vec.
 * @param incl          stride between consecutive elements in the `vec_l_g`.
 * @param vec_r_g       pointers to right vec.
 * @param incr          stride between consecutive elements in the `vec_r_g`.
 * @param results_g     dot product results.
 * @param batchcount    total count of dot products to compute.
 */
__global__ void psir_dot(const int* n,
                         double** vec_l_g,
                         int incl,
                         double** vec_r_g,
                         int incr,
                         double** results_g,
                         int batchcount);

} // namespace GintKernel
#endif // GINT_RHO_CUH
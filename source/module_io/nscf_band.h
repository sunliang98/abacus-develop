#ifndef NSCF_BAND_H
#define NSCF_BAND_H
#include "source_base/matrix.h"
#include "source_cell/klist.h"
#include "source_cell/parallel_kpoints.h"

namespace ModuleIO
{
/**
 * @brief calculate the band structure
 *
 * @param is spin index is = 0 or 1
 * @param out_band_dir directory to save the band structure
 * @param nband number of bands
 * @param fermie fermi energy
 * @param precision precision of the output
 * @param ekb eigenvalues of k points and bands
 * @param kv klist
 */
void nscf_band(const int& is,
               const std::string &eig_file, 
               const int& nband,
               const double& fermie,
               const int& precision,
               const ModuleBase::matrix& ekb,
               const K_Vectors& kv);
}

#endif

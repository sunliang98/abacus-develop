#ifndef DEEPKS_DESCRIPTOR_H
#define DEEPKS_DESCRIPTOR_H

#ifdef __MLALGO

#include "source_base/intarray.h"
#include "source_base/timer.h"
#include "source_cell/unitcell.h"

#include <torch/script.h>
#include <torch/torch.h>

namespace DeePKS_domain
{
//------------------------
// deepks_descriptor.cpp
//------------------------

// This file contains interfaces with libtorch,
// including loading of model and calculating gradients
// as well as subroutines that prints the results for checking

// The file contains 8 subroutines:
// 1. cal_descriptor : obtains descriptors which are eigenvalues of pdm
//       by calling torch::linalg::eigh
// 2. check_descriptor : prints descriptor for checking
// 3. cal_descriptor_equiv : calculates descriptor in equivalent version

/// Calculates descriptors
/// which are eigenvalues of pdm in blocks of I_n_l
void cal_descriptor(const int nat,
                    const int inlmax,
                    const std::vector<int>& inl2l,
                    const std::vector<torch::Tensor>& pdm,
                    std::vector<torch::Tensor>& descriptor,
                    const int des_per_atom);
/// print descriptors based on LCAO basis
void check_descriptor(const int inlmax,
                      const int des_per_atom,
                      const std::vector<int>& inl2l,
                      const UnitCell& ucell,
                      const std::string& out_dir,
                      const std::vector<torch::Tensor>& descriptor,
                      const int rank);

void cal_descriptor_equiv(const int nat,
                          const int des_per_atom,
                          const std::vector<torch::Tensor>& pdm,
                          std::vector<torch::Tensor>& descriptor);
} // namespace DeePKS_domain
#endif
#endif
#ifndef DEEPKS_ORBITAL_H
#define DEEPKS_ORBITAL_H

#ifdef __MLALGO

#include "source_base/complexmatrix.h"
#include "source_base/intarray.h"
#include "source_base/matrix.h"
#include "source_base/timer.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_estate/module_dm/density_matrix.h"

namespace DeePKS_domain
{
//------------------------
// deepks_orbital.cpp
//------------------------

// This file contains subroutines for calculating O_delta, i.e., corrections of the bandgap,
// which is defind as sum_mu,nu rho^{hl}_mu,nu <chi_mu|alpha>V(D)<alpha|chi_nu>
// where rho^{hl}_mu,nu = C_{L\mu}C_{L\nu} - C_{H\mu}C_{H\nu}, L for LUMO, H for HOMO

// There are 1 subroutine in this file:
// 1. cal_o_delta, which is used for O_delta calculation

template <typename TK, typename TH>
void cal_o_delta(const std::vector<TH>& dm_hl,
                 const std::vector<std::vector<TK>>& h_delta,
                 //  std::vector<double>& o_delta,
                 ModuleBase::matrix& o_delta,
                 const Parallel_Orbitals& pv,
                 const int nks,
                 const int nspin);
} // namespace DeePKS_domain

#endif
#endif

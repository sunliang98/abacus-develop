#ifndef SYMMETRY_RHO_H
#define SYMMETRY_RHO_H
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/module_symmetry/symmetry.h"
#include "source_estate/module_charge/charge.h"
#include "source_pw/hamilt_pwdft/parallel_grid.h"

class Symmetry_rho
{
  public:
    Symmetry_rho();
    ~Symmetry_rho();

    void begin(const int& spin_now,
               const Charge& CHR,
               const ModulePW::PW_Basis* pw,
               ModuleSymmetry::Symmetry& symm) const;

    void begin(const int& spin_now,
               double** rho,
               std::complex<double>** rhog,
               int ngmc,
               double** kin_r,
               const ModulePW::PW_Basis* pw,
               ModuleSymmetry::Symmetry& symm) const;

  private:
    // in real space:
    void psymm(double* rho_part,
               const ModulePW::PW_Basis* pw,
               Parallel_Grid& Pgrid,
               ModuleSymmetry::Symmetry& symm) const;
    // in reciprocal space:
    void psymmg(std::complex<double>* rhog_part,
                const ModulePW::PW_Basis* rho_basis,
                ModuleSymmetry::Symmetry& symm) const;
#ifdef __MPI
    void reduce_to_fullrhog(const ModulePW::PW_Basis* rho_basis,
                            std::complex<double>* rhogtot,
                            std::complex<double>* rhogin,
                            int* ig2isztot,
                            const int* ig2iszin,
                            int max_npw) const;
    void rhog_piece_to_all(const ModulePW::PW_Basis* rho_basis,
                           std::complex<double>* rhogtot,
                           std::complex<double>* rhog_part) const;
#endif
    void get_ixyz2ipw(const ModulePW::PW_Basis* rho_basis,
                      const int* ig2isztot,
                      const int* fftixy2is,
                      int* ixyz2ipw) const; //(ix, iy, iz) -> (ip, ig)
};

#endif

#ifndef VL_IN_PW_H 
#define VL_IN_PW_H 

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/matrix.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"

class pseudopot_cell_vl
{
public:

	pseudopot_cell_vl();
	~pseudopot_cell_vl();

    /**
     * @brief init local potential
     * 
     * @param rho_basis pw basis
     * @return this->vloc 
     */
    void init_vloc(const UnitCell& ucell,
                   const ModulePW::PW_Basis* rho_basis);

    ModuleBase::matrix vloc;   //(ntype,ngl),the local potential for each atom type(ntype,ngl)
	bool *numeric; //[ntype], =true

private:

	double *zp;   // (npsx),the charge of the pseudopotential

	void allocate(const UnitCell& ucell,
                  const int ngg);
    /**
     * @brief compute the coulomb potential in reciprocal space
     *        v(g) = -\frac{4pi}{V} * zp*e^2 / G^2
     */
    void vloc_coulomb(const UnitCell& ucell,
                      const double& zp, 
                      double* vloc_1d, 
                      const ModulePW::PW_Basis* rho_basis) const;
	// generate vloc for a particular atom type.
    void vloc_of_g(const int& msh,
                   const double* rab,
                   const double* r,
                   const double* vloc_at,
                   const double& zp,
                   double* vloc,
                   const UnitCell& ucell,
                   const ModulePW::PW_Basis* rho_basis) const;

    void print_vloc(const UnitCell& ucell,
                    const ModulePW::PW_Basis* rho_basis) const;
};

#endif // VL_IN_PW 

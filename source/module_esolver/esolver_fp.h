#ifndef ESOLVER_FP_H
#define ESOLVER_FP_H

#include "esolver.h"
#include "module_basis/module_pw/pw_basis.h"
#include "module_cell/module_symmetry/symmetry.h"
#include "module_elecstate/elecstate.h"
#include "module_elecstate/module_charge/charge_extra.h"
#include "module_hamilt_pw/hamilt_pwdft/structure_factor.h"

#include <fstream>

//! The First-Principles (FP) Energy Solver Class
/**
 * This class represents components that needed in 
 * first-principles energy solver, such as the plane
 * wave basis, the structure factors, and the k points.
 *
*/

namespace ModuleESolver
{
    class ESolver_FP : public ESolver
    {
    public:

        ModulePW::PW_Basis* pw_rho;

        /**
         * @brief same as pw_rho for ncpp. Here 'd' stands for 'dense'
         * dense grid for for uspp, used for ultrasoft augmented charge density.
         * charge density and potential are defined on dense grids,
         * but effective potential needs to be interpolated on smooth grids in order to compute Veff|psi>
         */
        ModulePW::PW_Basis* pw_rhod;
        ModulePW::PW_Basis_Big* pw_big; ///< [temp] pw_basis_big class

        //! Constructor
        ESolver_FP();

        //! Deconstructor
        virtual ~ESolver_FP();

        //! Initialize of the first-principels energy solver
        virtual void before_all_runners(const Input_para& inp, UnitCell& cell) override;

        virtual void init_after_vc(const Input_para& inp, UnitCell& cell);    // liuyu add 2023-03-09

        //! Electronic states
        elecstate::ElecState* pelec = nullptr;

        //! Electorn charge density
        Charge chr;

        //! Non-Self-Consistant Filed (NSCF) calculations
        virtual void nscf(){};

        //! Structure factors that used with plane-wave basis set
        Structure_Factor sf;

        //! K points in Brillouin zone
        K_Vectors kv;

      protected:
        //! Something to do after SCF iterations when SCF is converged or comes to the max iter step.
        virtual void after_scf(const int istep);

        //! Charge extrapolation
        Charge_Extra CE;

      private:
       
        //! Print charge density using FFT
        void print_rhofft(const Input_para& inp, std::ofstream &ofs);
    };
}

#endif

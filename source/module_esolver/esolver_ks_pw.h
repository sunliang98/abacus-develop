#ifndef ESOLVER_KS_PW_H
#define ESOLVER_KS_PW_H
#include "./esolver_ks.h"
#include "module_hamilt_pw/hamilt_pwdft/operator_pw/velocity_pw.h"
#include "module_psi/psi_initializer.h"
#include <memory>
#include <module_base/macros.h>

// #include "Basis_PW.h"
// #include "Estate_PW.h"
// #include "Hamilton_PW.h"
// #include "H2E_pw.h"


namespace ModuleESolver
{

template<typename T, typename Device = psi::DEVICE_CPU>
class ESolver_KS_PW : public ESolver_KS<T, Device>
{
	private:

		using Real = typename GetTypeReal<T>::type;

	public:

        ESolver_KS_PW();

        ~ESolver_KS_PW();

        void init(Input& inp, UnitCell& cell) override;

        void init_after_vc(Input& inp, UnitCell& cell) override;

        double cal_energy() override;

        void cal_force(ModuleBase::matrix& force) override;

        void cal_stress(ModuleBase::matrix& stress) override;

        virtual void hamilt2density(const int istep, const int iter, const double ethr) override;

        virtual void hamilt2estates(const double ethr) override;

        virtual void nscf() override;

        void post_process() override;

        /**
         * @brief calculate Onsager coefficients Lmn(\omega) and conductivities with Kubo-Greenwood formula
         * 
         * @param fwhmin FWHM for delta function
         * @param smear_type 1: Gaussian, 2: Lorentzian
         * @param wcut cutoff \omega for Lmn(\omega)
         * @param dw_in \omega step
         * @param dt_in time step
         * @param wg wg(ik,ib) occupation for the ib-th band in the ik-th kpoint
         */
        void KG(const int& smear_type,
                const double fwhmin,
                const double wcut,
                const double dw_in,
                const double dt_in,
                ModuleBase::matrix& wg);

        /**
         * @brief calculate the response function Cmn(t) for currents
         * 
         * @param ik k point
         * @param nt number of steps of time
         * @param dt time step
         * @param decut ignore dE which is larger than decut
         * @param wg wg(ik,ib) occupation for the ib-th band in the ik-th kpoint
         * @param velop velocity operator
         * @param ct11 C11(t)
         * @param ct12 C12(t)
         * @param ct22 C22(t)
         */
        void jjcorr_ks(const int ik,
                       const int nt,
                       const double dt,
                       const double decut,
                       ModuleBase::matrix& wg,
                       hamilt::Velocity& velop,
                       double* ct11,
                       double* ct12,
                       double* ct22);

      protected:

        virtual void before_scf(const int istep) override;

        virtual void iter_init(const int istep, const int iter) override;

        virtual void update_pot(const int istep, const int iter) override;

        virtual void iter_finish(const int iter) override;

        virtual void after_scf(const int istep) override;

        virtual void others(const int istep)override;

        //temporary, this will be removed in the future;
        //Init Global class
        void Init_GlobalC(Input& inp, UnitCell& cell);

        /// @brief calculate conductivities from j-j correlation function
        void calcondw(const int nt,
                      const double dt,
                      const int& smear_type,
                      const double fwhmin,
                      const double wcut,
                      const double dw_in,
                      double* ct11,
                      double* ct12,
                      double* ct22);

        /// @brief allocate psi_init the new psi_initializer
        void allocate_psi_init();

        /// @brief initialize psi
        void initialize_psi();

    protected:

        //! hide the psi in ESolver_KS for tmp use
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>* psi = nullptr;

    private:

        // psi_initializer<T, Device>* psi_init = nullptr;
        // change to use smart pointer to manage the memory, and avoid memory leak
        // while the std::make_unique() is not supported till C++14, 
        // so use the new and std::unique_ptr to manage the memory, but this makes new-delete not symmetric
        std::unique_ptr<psi_initializer<T, Device>> psi_init;

        Device * ctx = {};

        psi::AbacusDevice_t device = {};

        psi::Psi<T, Device>* kspw_psi = nullptr;

        psi::Psi<std::complex<double>, Device>* __kspw_psi = nullptr;

        using castmem_2d_d2h_op = psi::memory::cast_memory_op<std::complex<double>, T, psi::DEVICE_CPU, Device>;
    };
}  // namespace ModuleESolver
#endif

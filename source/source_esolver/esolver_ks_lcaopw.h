#ifndef ESOLVER_KS_LIP_H
#define ESOLVER_KS_LIP_H
#include "source_esolver/esolver_ks_pw.h"
#include "source_hsolver/hsolver_lcaopw.h"

#ifdef __EXX
#include "source_lcao/module_ri/exx_lip.h"
#endif
namespace ModuleESolver
{

    template <typename T>
    class ESolver_KS_LIP : public ESolver_KS_PW<T, base_device::DEVICE_CPU>
    {
    private:
        using Real = typename GetTypeReal<T>::type;

    public:
        ESolver_KS_LIP();

        ~ESolver_KS_LIP();

        void before_all_runners(UnitCell& ucell, const Input_para& inp) override;
        void after_all_runners(UnitCell& ucell) override;

        virtual void before_scf(UnitCell& ucell, const int istep) override;

      protected:
        virtual void iter_init(UnitCell& ucell, const int istep, const int iter) override;
        virtual void iter_finish(UnitCell& ucell, const int istep, int& iter, bool& conv_esolver) override;

        /// All the other interfaces except this one are the same as ESolver_KS_PW.
        virtual void hamilt2rho_single(UnitCell& ucell,
                                           const int istep,
                                           const int iter,
                                           const double ethr) override;

        virtual void allocate_hamilt(const UnitCell& ucell) override;
        virtual void deallocate_hamilt() override;
        
        psi::Psi<T, base_device::DEVICE_CPU>* psi_local = nullptr; ///< psi for all local NAOs

#ifdef __EXX
        std::unique_ptr<Exx_Lip<T>> exx_lip;
        int two_level_step = 0;
#endif

    };
} // namespace ModuleESolver
#endif

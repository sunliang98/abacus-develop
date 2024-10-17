#ifndef ESOLVER_KS_LIP_H
#define ESOLVER_KS_LIP_H
#include "module_esolver/esolver_ks_pw.h"
#include "module_hsolver/hsolver_lcaopw.h"

#ifdef __EXX
#include "module_ri/exx_lip.h"
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

        /// All the other interfaces except this one are the same as ESolver_KS_PW.
        virtual void hamilt2density(const int istep, const int iter, const double ethr) override;

        void before_all_runners(const Input_para& inp, UnitCell& cell) override;
        void after_all_runners()override;

    protected:
      virtual void iter_init(const int istep, const int iter) override;
      virtual void iter_finish(int& iter) override;

      virtual void allocate_hamilt() override;
      virtual void deallocate_hamilt() override;

#ifdef __EXX
        std::unique_ptr<Exx_Lip<T>> exx_lip;
        int two_level_step = 0;
#endif

    };
} // namespace ModuleESolver
#endif

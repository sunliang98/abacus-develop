#ifndef RELAX_DRIVER_H
#define RELAX_DRIVER_H

#include "source_cell/unitcell.h"
#include "source_esolver/esolver.h"
#include "source_esolver/esolver_ks.h"
#include "relax_sync.h"
#include "relax_nsync.h"
#include "bfgs.h"
class Relax_Driver
{

  public:
    Relax_Driver(){};
    ~Relax_Driver(){};

    void relax_driver(ModuleESolver::ESolver* p_esolver, UnitCell& ucell);

  private:
    // mohan add 2021-01-28
    // mohan moved this variable from electrons.h to relax_driver.h
    int istep = 0;
    double etot = 0;

    // new relaxation method
    Relax rl;

    // old relaxation method
    Relax_old rl_old;

    BFGS bfgs_trad;


};

#endif

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "gint_k.h"
#include "source_basis/module_ao/ORB_read.h"
#include "grid_technique.h"
#include "source_base/ylm.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "source_base/blas_connector.h"
#include "source_base/timer.h"
#include "source_base/array_pool.h"
#include "gint_tools.h"
#include "source_base/memory.h"
#include "source_lcao/module_gint/grid_technique.h"


void Gint::cal_meshball_tau(
	const int na_grid,
	int* block_index,
	int* vindex,
	double** dpsix,
	double** dpsiy,
	double** dpsiz,
	double** dpsix_dm,
	double** dpsiy_dm,
	double** dpsiz_dm,
	double* rho)
{		
	const int inc = 1;
	// sum over mu to get density on grid
	for(int ib=0; ib<this->bxyz; ++ib)
	{
		double rx=ddot_(&block_index[na_grid], dpsix[ib], &inc, dpsix_dm[ib], &inc);
		double ry=ddot_(&block_index[na_grid], dpsiy[ib], &inc, dpsiy_dm[ib], &inc);
		double rz=ddot_(&block_index[na_grid], dpsiz[ib], &inc, dpsiz_dm[ib], &inc);
		const int grid = vindex[ib];
		rho[ grid ] += rx + ry + rz;
	}
}
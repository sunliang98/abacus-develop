#ifndef EXX_ABFS_JLE_H
#define EXX_ABFS_JLE_H

#include "exx_abfs.h"

#include <vector>
#include "source_cell/unitcell.h"
#include "../../source_basis/module_ao/ORB_read.h"

class Exx_Abfs::Jle
{
public:
	
	std::vector<
		std::vector<
			std::vector<
				Numerical_Orbital_Lm>>> jle;

	void init_jle(const double kmesh_times, 
				  const UnitCell& ucell,
				  const LCAO_Orbitals& orb);

	static bool generate_matrix;
	static int Lmax;
	static double Ecut_exx;
	static double tolerence;
};

#endif	// EXX_ABFS_JLE_H

#include "grid_meshball.h"
#include "source_base/memory.h"
#include "module_parameter/parameter.h"

Grid_MeshBall::Grid_MeshBall()
{
}

Grid_MeshBall::~Grid_MeshBall()
{
}

void Grid_MeshBall::init_meshball()
{	
	ModuleBase::TITLE("Grid_MeshBall","init_meshball");

    // init meshball_radius, generally the value
    // is same as orbital_rmax, of course you can
    // incrase meshball_radius, but there will be
    // no atoms in the added bigcells.
    // (in case subcell are too many).
	this->meshball_radius = this->orbital_rmax;

	// select a ball in a cubic.
	double pos[3];
	double r2=0.0;

	//------------------------------------------------------------------
	// const double rcut2 = this->meshball_radius * this->meshball_radius;
	// qianrui fix a bug and add 0.001 2022-4-30
	// Sometimes r2 is equal to rcut2, for example they are 36.
	// However, r2 is either 35.99.. or 36.0..001， which makes  count != this->meshball_ncells
	// and segment fault.
	// I do not know how to solve it and this may occurs in somewhere else in ABACUS.
	// May some genius can give a better solution.
	//------------------------------------------------------------------
	const double rcut2 = this->meshball_radius * this->meshball_radius + 0.001;
	
	//-------------------------------------------------------------------
	// calculate twice, the first time find the number of mesh points,
	// then allocate array and save each bigcell's cartesian coordinate.
	// plus one because we need to cover atom spillage.
	// meshball_ncells: How many cells in mesh ball.
	//-------------------------------------------------------------------
	this->meshball_ncells = 0;
	for(int i=-dxe; i<dxe+1; i++) // mohan fix bug 2009-10-21, range should be [-dxe,dxe]
	{
		for(int j=-dye; j<dye+1; j++)
		{
			for(int k=-dze; k<dze+1; k++)
			{
				// caclculate the std::vector away from 'zero point'.
				for(int ip=0; ip<3; ip++)
				{
					pos[ip] = i*bigcell_vec1[ip]+j*bigcell_vec2[ip]+k*bigcell_vec3[ip];
				}
				r2 = this->deal_with_atom_spillage( pos );
				//r2 = pos[0]*pos[0]+pos[1]*pos[1]+pos[2]*pos[2];
	
				// calculate the distance.
				if( r2 < rcut2 )
				{
					++meshball_ncells;
				} 
			}
		}
	}
	if(PARAM.inp.test_gridt) {ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "how many cells in meshball",this->meshball_ncells);
}

	// prepare for the second calculation.
	this->meshball_positions = std::vector<std::vector<double>>(meshball_ncells, std::vector<double>(3, 0.0));
	ModuleBase::Memory::record("meshball_pos", sizeof(double) * meshball_ncells*3);
    this->index_ball = std::vector<int>(meshball_ncells);
	ModuleBase::Memory::record("index_ball", sizeof(int) * meshball_ncells);

	// second time.
	int count = 0;
	for(int i=-dxe; i<this->dxe+1; i++)
	{
		for(int j=-dye; j<this->dye+1; j++)
		{
			for(int k=-dze; k<this->dze+1; k++)
			{
				// caclculate the std::vector away from 'zero point'.
				// change to cartesian coordinates.
				for(int ip=0; ip<3; ip++)
				{
					pos[ip] = i*bigcell_vec1[ip]+j*bigcell_vec2[ip]+k*bigcell_vec3[ip];
				}
				r2 = this->deal_with_atom_spillage( pos );

				// calculate the distance.
				if( r2 < rcut2 )
				{
					for(int ip=0; ip<3; ip++)
					{
						this->meshball_positions[count][ip] = pos[ip];
					}

					// record each position.
					this->index_ball[count] = k + j * this->nze + i * this->nye * this->nze;
					++count;
				} 
			}
		}
	}

	assert(count == this->meshball_ncells);
	return;
}

double Grid_MeshBall::deal_with_atom_spillage(const double *pos)
{
	double dx;
	double r2 = 100000;
	double *cell=new double[3];
	
	for(int i=-1; i<=1; i++)
	{
		for(int j=-1; j<=1; j++)
		{
			for(int k=-1; k<=1; k++)
			{
				dx = 0.0;
				for(int ip=0; ip<3; ip++)
				{
					// change to cartesian coordinates.	
					cell[ip] = i*this->bigcell_vec1[ip] +
						j*this->bigcell_vec2[ip] +
						k*this->bigcell_vec3[ip];
					dx += (cell[ip] - pos[ip]) * (cell[ip] - pos[ip]);
				}
				r2 = std::min(dx, r2);
			}
		}
	}
	delete[] cell;
	return r2;
}



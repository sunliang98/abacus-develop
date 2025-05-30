# Brief Introduction of the Input Files

The following files are the central input files for ABACUS. Before executing the program, please make sure these files are prepared and stored in the working directory. Here we give some simple descriptions. For more details, users should consult the Advanced session.

## *INPUT*

The `INPUT` file contains parameters that control the type of calculation as well as a variety of settings.

Below is an example `INPUT` file with some of the most important parameters that need to be set:

```plaintext
INPUT_PARAMETERS
suffix                  MgO  # the output files will be in OUT.{suffix} directory 
pseudo_dir              ./   # where the pseudopotential for each element is
orbital_dir             ./   # where the orbital file for each element is
ecutwfc                 100  # in Rydberg
scf_thr                 1e-6 # dimensionless for LCAO, Rydberg for PW. See documents for details.
basis_type              lcao # lcao or pw
calculation             scf  # this is the key parameter telling abacus to do a scf calculation
out_chg                 0    # only output binary charge file for restart
```

The parameter list always starts with key word `INPUT_PARAMETERS`. Any content before `INPUT_PARAMETERS` will be ignored.

Any line starting with `#` or `/` will also be ignored.

Each parameter value is provided by specifying the name of the input variable
and then putting the value after the name, separated by one or more blank characters(space or tab). The following characters (> 150) in the same line will be neglected.

Depending on the input variable, the value may be an integer, a real number or a string. The parameters can be given in any order, but only one parameter should be given per line.

Furthermore, if a given parameter name appeared more than once in the input file, only the last value will be taken.

> **Note:** if a parameter name is not recognized by the program, the program will stop with an error message.

In the above example, the meanings of the parameters are:

- `suffix` : the name of the system, default `ABACUS`, and output files will be in OUT.{suffix} directory. 
- `pseudo_dir` : the directory where pseudopotential files are provided.
- `orbital_dir` : the directory where orbital files are provided.
- `ecutwfc` : the plane-wave energy cutoff for the wave function expansion (UNIT: Rydberg).
- `scf_thr` : the threshold for the convergence of charge density (UNIT: Rydberg for PW, dimensionless for LCAO), we recommend `1e-7` for LCAO and `1e-9` for PW basis.
- `basis_type` : the type of basis set for expanding the electronic wave functions, one can set lcao or pw.
- `calculation` : the type of calculation to be performed by ABACUS
- `out_chg` : setting for output the charge density in real space grid, -1 for no output, 0 for binary output, 1 for binary and cube output.

For a complete list of input parameters, please consult this [instruction](../advanced/input_files/input-main.md).

> **Note:** Users cannot change the filename “INPUT” to other names. Boolean paramerters such as `out_chg` can be set by using `True` and `False`, `1` and `0`, or `T` and `F`. It is case insensitive so that other preferences such as `true` and `false`, `TRUE` and `FALSE`, and `t` and `f` for setting boolean values are also supported. Specifically for the `out_chg`, `-1` option is also available, which means turn off the checkpoint of charge density in binary (always dumped in `OUT.{suffix}`, whose name ends with `CHARGE-DENSITY.restart`). Some parameters controlling the output also support a second option to control the output precision, e.g., `out_chg 1 8` will output the charge density on realspace grid with 8 digits after the decimal point.

## *STRU*

The structure file contains structural information about the system, e.g., lattice constant, lattice vectors, and positions of the atoms within a unit cell. The positions can be given either in direct or Cartesian coordinates.

An example of the `STRU` file is given as follows :

```plaintext
#This is the atom file containing all the information
#about the lattice structure.

ATOMIC_SPECIES
Mg 24.305  Mg_ONCV_PBE-1.0.upf  # element name, atomic mass, pseudopotential file
O  15.999 O_ONCV_PBE-1.0.upf

NUMERICAL_ORBITAL
Mg_gga_8au_100Ry_4s2p1d.orb
O_gga_8au_100Ry_2s2p1d.orb

LATTICE_CONSTANT
1.889726126 # 1.0 Ang = 1/a_0 = 1/0.529177210544
# Bohr radius ref: https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0

LATTICE_VECTORS
4.25648 0.00000 0.00000  
0.00000 4.25648 0.00000
0.00000 0.00000 4.25648

ATOMIC_POSITIONS
Direct                  #Cartesian(Unit is LATTICE_CONSTANT)
Mg                      #Name of element        
0.0                     #Magnetic for this element.
4                       #Number of atoms
0.0  0.0  0.0  0 0 0    #x,y,z, move_x, move_y, move_z
0.0  0.5  0.5  0 0 0    #x,y,z, move_x, move_y, move_z
0.5  0.0  0.5  0 0 0    #x,y,z, move_x, move_y, move_z
0.5  0.5  0.0  0 0 0    #x,y,z, move_x, move_y, move_z
O                       #Name of element        
0.0                     #Magnetic for this element.
4                       #Number of atoms
0.5  0.0  0.0  0 0 0    #x,y,z, move_x, move_y, move_z
0.5  0.5  0.5  0 0 0    #x,y,z, move_x, move_y, move_z
0.0  0.0  0.5  0 0 0    #x,y,z, move_x, move_y, move_z
0.0  0.5  0.0  0 0 0    #x,y,z, move_x, move_y, move_z
```

> **Note:** users may choose a different name for their structure file using the keyword `stru_file`. The order of the pseudopotential file list and the numerical orbital list (if LCAO is applied) MUST be consistent with that of the atomic type given in `ATOMIC_POSITIONS`.

For a more detailed description of STRU file, please consult [here](../advanced/input_files/stru.md).

## *KPT*

This file contains information of the kpoint grid setting for the Brillouin zone sampling.

An example of the `KPT` file is given below:

```plaintext
K_POINTS
0 
Gamma
4 4 4 0 0 0
```

> **Note:** users may choose a different name for their k-point file using keyword `kpoint_file`

For a more detailed description, please consult [here](../advanced/input_files/kpt.md).

- The pseudopotential files

    Norm-conserving pseudopotentials are used in ABACUS, in the UPF file format.The filename of each element’s pseudopotential needs to be specified in the STRU file, together with the directory of the pseudopotential files unless they are already present in the working directory.

    More information on pseudopotentials is given [here](../advanced/pp_orb.md#pseudopotentials).

- The numerical orbital files

    This part is only required in LCAO calculations.
    The filename for each element’s numerical orbital basis needs to be specified in the STRU file, together with the directory of the orbital files unless they are already present in the working directory.
    ABACUS provides numerical atomic basis sets of different accuracy levels for most elements commonly used. Users can download these basis sets from the [website](http://abacus.ustc.edu.cn/pseudo/list.htm). Moreover, users can generate numerical atomic orbitals by themselves, and the procedure is provided in this [short introduction](../advanced/pp_orb.md#generating-atomic-orbital-bases).

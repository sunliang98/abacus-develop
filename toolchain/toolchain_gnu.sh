#!/bin/bash
#SBATCH -J install
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -o compile.log
#SBATCH -e compile.err

# JamesMisaka in 2023-09-16
# install abacus dependency by gnu-toolchain
# one can use mpich or openmpi.
# openmpi will be faster, but not compatible in some cases.
# libtorch and libnpy are for deepks support, which can be =no
# if you want to run EXX calculation, you should set --with-libri=install
# mpich (and intel toolchain) is recommended for EXX support

./install_abacus_toolchain.sh --with-openmpi=install \
--with-intel=no --with-gcc=system \
--with-cmake=install \
--with-scalapack=install \
--with-libxc=install \
--with-fftw=install \
--with-elpa=install \
--with-cereal=install \
--with-rapidjson=install \
--with-libtorch=no \
--with-libnpy=no \
--with-libri=no \
--with-libcomm=no \
| tee compile.log
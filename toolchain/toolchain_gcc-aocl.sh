#!/bin/bash
#SBATCH -J install
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -o compile.log
#SBATCH -e compile.err

# JamesMisaka in 2025-05-05
# install abacus dependency by gcc-aocl toolchain
# openmpi is recommended to use
# libtorch and libnpy are for deepks support, which can be =no
# if you want to run EXX calculation, you should set --with-libri=install

# gpu-lcao supporting modify: CUDA_PATH and --enable-cuda
# export CUDA_PATH=/usr/local/cuda

./install_abacus_toolchain.sh \
--with-gcc=system \
--math-mode=aocl \
--with-amd=no \
--with-intel=no \
--with-openmpi=install \
--with-cmake=install \
--with-scalapack=system \
--with-libxc=install \
--with-fftw=system \
--with-elpa=install \
--with-cereal=install \
--with-rapidjson=install \
--with-libtorch=no \
--with-libnpy=no \
--with-libri=no \
--with-libcomm=no \
--with-4th-openmpi=no \
| tee compile.log
# to use openmpi-version4: set --with-4th-openmpi=yes
#!/bin/bash -e

# TODO: Review and if possible fix shellcheck errors.
# shellcheck disable=all

# Last Update in 2023-0901

[ "${BASH_SOURCE[0]}" ] && SCRIPT_NAME="${BASH_SOURCE[0]}" || SCRIPT_NAME=$0
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_NAME")/.." && pwd -P)"

fftw_ver="3.3.10"
fftw_sha256="56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"
fftw_pkg="fftw-${fftw_ver}.tar.gz"

source "${SCRIPT_DIR}"/common_vars.sh
source "${SCRIPT_DIR}"/tool_kit.sh
source "${SCRIPT_DIR}"/signal_trap.sh
source "${INSTALLDIR}"/toolchain.conf
source "${INSTALLDIR}"/toolchain.env

[ -f "${BUILDDIR}/setup_fftw" ] && rm "${BUILDDIR}/setup_fftw"

FFTW_CFLAGS=''
FFTW_LDFLAGS=''
FFTW_LIBS=''
! [ -d "${BUILDDIR}" ] && mkdir -p "${BUILDDIR}"
cd "${BUILDDIR}"

case "$with_fftw" in
  __INSTALL__)
    require_env MPI_LIBS
    echo "==================== Installing FFTW ===================="
    pkg_install_dir="${INSTALLDIR}/fftw-${fftw_ver}"
    #pkg_install_dir="${HOME}/lib/fftw/${fftw_ver}-gcc8"
    install_lock_file="$pkg_install_dir/install_successful"

    if verify_checksums "${install_lock_file}"; then
      echo "fftw-${fftw_ver} is already installed, skipping it."
    else
      if [ -f ${fftw_pkg} ]; then
        echo "${fftw_pkg} is found"
      else
        #download_pkg_from_ABACUS_org "${fftw_sha256}" "${fftw_pkg}"
        url="http://www.fftw.org/${fftw_pkg}"
        download_pkg_from_url "${fftw_sha256}" "${fftw_pkg}" "${url}"
      fi
    if [ "${PACK_RUN}" = "__TRUE__" ]; then
      echo "--pack-run mode specified, skip installation"
    else
      echo "Installing from scratch into ${pkg_install_dir}"
      [ -d fftw-${fftw_ver} ] && rm -rf fftw-${fftw_ver}
      tar -xzf ${fftw_pkg}
      cd fftw-${fftw_ver}
      FFTW_FLAGS="--enable-openmp --enable-shared"
      # fftw has mpi support but not compiled by default. so compile it if we build with mpi.
      # it will create a second library to link with if needed
      [ "${MPI_MODE}" != "no" ] && FFTW_FLAGS="--enable-mpi ${FFTW_FLAGS}"
      if [ "${TARGET_CPU}" = "native" ]; then
        if [ -f /proc/cpuinfo ]; then
          grep '\bavx\b' /proc/cpuinfo 1> /dev/null && FFTW_FLAGS="${FFTW_FLAGS} --enable-avx"
          grep '\bavx2\b' /proc/cpuinfo 1> /dev/null && FFTW_FLAGS="${FFTW_FLAGS} --enable-avx2"
          grep '\bavx512f\b' /proc/cpuinfo 1> /dev/null && FFTW_FLAGS="${FFTW_FLAGS} --enable-avx512"
        fi
      fi
      # ABACUS need float version and double version fftw at the same time
      # install float version fftw
      echo "install float version fftw"
      ./configure --prefix=${pkg_install_dir} --libdir="${pkg_install_dir}/lib" ${FFTW_FLAGS} --enable-float \
        > configure.log 2>&1 || tail -n ${LOG_LINES} configure.log
      make -j $(get_nprocs) > make.log 2>&1 || tail -n ${LOG_LINES} make.log
      make install > install.log 2>&1 || tail -n ${LOG_LINES} install.log
       # install double version fftw
      echo "clean"
      make distclean > /dev/null 2>&1 || true
      echo "install double version fftw"
      ./configure --prefix=${pkg_install_dir} --libdir="${pkg_install_dir}/lib" ${FFTW_FLAGS} \
        > configure.log 2>&1 || tail -n ${LOG_LINES} configure.log
      make -j $(get_nprocs) > make.log 2>&1 || tail -n ${LOG_LINES} make.log
      make install > install.log 2>&1 || tail -n ${LOG_LINES} install.log
      cd ..
      write_checksums "${install_lock_file}" "${SCRIPT_DIR}/stage3/$(basename ${SCRIPT_NAME})"
    fi
    fi
    FFTW_CFLAGS="-I'${pkg_install_dir}/include'"
    FFTW_LDFLAGS="-L'${pkg_install_dir}/lib' -Wl,-rpath,'${pkg_install_dir}/lib'"
    ;;
  __SYSTEM__)
    echo "==================== Finding FFTW from system paths ===================="
    check_lib -lfftw3 "FFTW"
    check_lib -lfftw3_omp "FFTW"
    [ "${MPI_MODE}" != "no" ] && check_lib -lfftw3_mpi "FFTW"
    add_include_from_paths FFTW_CFLAGS "fftw3.h" FFTW_INC ${INCLUDE_PATHS}
    add_lib_from_paths FFTW_LDFLAGS "libfftw3.*" ${LIB_PATHS}
    ;;
  __DONTUSE__)
    # Nothing to do
    ;;
  *)
    echo "==================== Linking FFTW to user paths ===================="
    pkg_install_dir="$with_fftw"
    check_dir "${pkg_install_dir}/lib"
    check_dir "${pkg_install_dir}/include"
    FFTW_CFLAGS="-I'${pkg_install_dir}/include'"
    FFTW_LDFLAGS="-L'${pkg_install_dir}/lib' -Wl,-rpath,'${pkg_install_dir}/lib'"
    ;;
esac
if [ "$with_fftw" != "__DONTUSE__" ]; then
  [ "$MPI_MODE" != "no" ] && FFTW_LIBS="IF_MPI(-lfftw3_mpi|)"
  FFTW_LIBS+="-lfftw3 -lfftw3_omp"
  if [ "$with_fftw" != "__SYSTEM__" ]; then
    cat << EOF > "${BUILDDIR}/setup_fftw"
prepend_path LD_LIBRARY_PATH "$pkg_install_dir/lib"
prepend_path LD_RUN_PATH "$pkg_install_dir/lib"
prepend_path LIBRARY_PATH "$pkg_install_dir/lib"
prepend_path CPATH "$pkg_install_dir/include"
prepend_path PKG_CONFIG_PATH "$pkg_install_dir/lib/pkgconfig"
prepend_path CMAKE_PREFIX_PATH "$pkg_install_dir"
export LD_LIBRARY_PATH="$pkg_install_dir/lib":\${LD_LIBRARY_PATH}
export LD_RUN_PATH="$pkg_install_dir/lib":\${LD_RUN_PATH}
export LIBRARY_PATH="$pkg_install_dir/lib":\${LIBRARY_PATH}
export CPATH="$pkg_install_dir/include":\${CPATH}
export PKG_CONFIG_PATH="$pkg_install_dir/lib/pkgconfig":\${PKG_CONFIG_PATH}
export CMAKE_PREFIX_PATH="$pkg_install_dir":\${CMAKE_PREFIX_PATH}
EOF
  fi
  # we may also want to cover FFT_SG
  cat << EOF >> "${BUILDDIR}/setup_fftw"
export FFTW3_INCLUDES="${FFTW_CFLAGS}"
export FFTW3_LIBS="${FFTW_LIBS}"
export FFTW_CFLAGS="${FFTW_CFLAGS}"
export FFTW_LDFLAGS="${FFTW_LDFLAGS}"
export FFTW_LIBS="${FFTW_LIBS}"
export CP_DFLAGS="\${CP_DFLAGS} -D__FFTW3 IF_COVERAGE(IF_MPI(|-U__FFTW3)|)"
export CP_CFLAGS="\${CP_CFLAGS} ${FFTW_CFLAGS}"
export CP_LDFLAGS="\${CP_LDFLAGS} ${FFTW_LDFLAGS}"
export CP_LIBS="${FFTW_LIBS} \${CP_LIBS}"
export FFTW_ROOT=${FFTW_ROOT:-${pkg_install_dir}}
export FFTW3_ROOT=${pkg_install_dir}
EOF
  cat "${BUILDDIR}/setup_fftw" >> $SETUPFILE
fi
cd "${ROOTDIR}"

# ----------------------------------------------------------------------
# Suppress reporting of known leaks
# ----------------------------------------------------------------------
cat << EOF >> ${INSTALLDIR}/valgrind.supp
{
   <BuggyFFTW3>
   Memcheck:Addr32
   fun:cdot
   ...
   fun:invoke_solver
   fun:search0
}
EOF

load "${BUILDDIR}/setup_fftw"
write_toolchain_env "${INSTALLDIR}"

report_timing "fftw"

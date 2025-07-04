#!/bin/bash -e

# TODO: Review and if possible fix shellcheck errors.
# shellcheck disable=all

# Last Update in 2024-0811

[ "${BASH_SOURCE[0]}" ] && SCRIPT_NAME="${BASH_SOURCE[0]}" || SCRIPT_NAME=$0
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_NAME")/.." && pwd -P)"

openblas_ver="0.3.29" # Keep in sync with get_openblas_arch.sh
openblas_sha256="38240eee1b29e2bde47ebb5d61160207dc68668a54cac62c076bb5032013b1eb"
openblas_pkg="OpenBLAS-${openblas_ver}.tar.gz"

source "${SCRIPT_DIR}"/common_vars.sh
source "${SCRIPT_DIR}"/tool_kit.sh
source "${SCRIPT_DIR}"/signal_trap.sh
source "${INSTALLDIR}"/toolchain.conf
source "${INSTALLDIR}"/toolchain.env

[ -f "${BUILDDIR}/setup_openblas" ] && rm "${BUILDDIR}/setup_openblas"

OPENBLAS_CFLAGS=""
OPENBLAS_LDFLAGS=""
OPENBLAS_LIBS=""
OPENBLAS_ROOT=""
! [ -d "${BUILDDIR}" ] && mkdir -p "${BUILDDIR}"
cd "${BUILDDIR}"

case "${with_openblas}" in
  __INSTALL__)
    echo "==================== Installing OpenBLAS ===================="
    pkg_install_dir="${INSTALLDIR}/openblas-${openblas_ver}"
    #pkg_install_dir="${HOME}/lib/openblas/${openblas_ver}-gcc8"
    install_lock_file="$pkg_install_dir/install_successful"
    if verify_checksums "${install_lock_file}"; then
      echo "openblas-${openblas_ver} is already installed, skipping it."
    else
      if [ -f ${openblas_pkg} ]; then
        echo "${openblas_pkg} is found"
      else
        #download_pkg_from_ABACUS_org "${openblas_sha256}" "${openblas_pkg}"
        # using codeload.github
        url="https://codeload.github.com/OpenMathLib/OpenBLAS/tar.gz/v${openblas_ver}"
        download_pkg_from_url "${openblas_sha256}" "${openblas_pkg}" "${url}"
      fi
    if [ "${PACK_RUN}" = "__TRUE__" ]; then
      echo "--pack-run mode specified, skip installation"
    else
      echo "Installing from scratch into ${pkg_install_dir}"
      [ -d OpenBLAS-${openblas_ver} ] && rm -rf OpenBLAS-${openblas_ver}
      tar -zxf ${openblas_pkg}
      cd OpenBLAS-${openblas_ver}

      # First attempt to make openblas using auto detected
      # TARGET, if this fails, then make with forced
      # TARGET=NEHALEM
      #
      # wrt NUM_THREADS=64: this is what the most common Linux distros seem to choose atm
      #                     for a good compromise between memory usage and scalability
      #
      # Unfortunately, NO_SHARED=1 breaks ScaLAPACK build.
      case "${TARGET_CPU}" in
        "generic")
          TARGET="NEHALEM"
          ;;
        "native")
          TARGET=${OPENBLAS_LIBCORE}
          ;;
        "broadwell" | "skylake")
          TARGET="HASWELL"
          ;;
        "skylake-avx512")
          TARGET="SKYLAKEX"
          ;;
        *)
          TARGET=${TARGET_CPU}
          ;;
      esac
      TARGET=$(echo ${TARGET} | tr '[:lower:]' '[:upper:]')
      echo "Installing OpenBLAS library for target ${TARGET}"
      (
        make -j $(get_nprocs) \
          MAKE_NB_JOBS=0 \
          TARGET=${TARGET} \
          NUM_THREADS=64 \
          USE_THREAD=1 \
          USE_OPENMP=1 \
          NO_AFFINITY=1 \
          CC="${CC}" \
          FC="${FC}" \
          PREFIX="${pkg_install_dir}" \
          > make.log 2>&1 || tail -n ${LOG_LINES} make.log
      ) || (
        make -j $(get_nprocs) \
          MAKE_NB_JOBS=0 \
          TARGET=NEHALEM \
          NUM_THREADS=64 \
          USE_THREAD=1 \
          USE_OPENMP=1 \
          NO_AFFINITY=1 \
          CC="${CC}" \
          FC="${FC}" \
          PREFIX="${pkg_install_dir}" \
          > make.nehalem.log 2>&1 || tail -n ${LOG_LINES} make.nehalem.log
      )
      make -j $(get_nprocs) \
        MAKE_NB_JOBS=0 \
        TARGET=${TARGET} \
        NUM_THREADS=64 \
        USE_THREAD=1 \
        USE_OPENMP=1 \
        NO_AFFINITY=1 \
        CC="${CC}" \
        FC="${FC}" \
        PREFIX="${pkg_install_dir}" \
        install > install.log 2>&1 || tail -n ${LOG_LINES} install.log
      cd ..
      write_checksums "${install_lock_file}" "${SCRIPT_DIR}/stage2/$(basename ${SCRIPT_NAME})"
    fi
    fi
    OPENBLAS_CFLAGS="-I'${pkg_install_dir}/include'"
    OPENBLAS_LDFLAGS="-L'${pkg_install_dir}/lib' -Wl,-rpath,'${pkg_install_dir}/lib'"
    OPENBLAS_ROOT="${pkg_install_dir}"
    OPENBLAS_LIBS="-lopenblas"
    ;;
  __SYSTEM__)
    echo "==================== Finding LAPACK from system paths ===================="
    # assume that system openblas is threaded
    check_lib -lopenblas "OpenBLAS"
    OPENBLAS_LIBS="-lopenblas"
    # detect separate omp builds
    check_lib -lopenblas_openmp 2> /dev/null && OPENBLAS_LIBS="-lopenblas_openmp"
    check_lib -lopenblas_omp 2> /dev/null && OPENBLAS_LIBS="-lopenblas_omp"
    add_include_from_paths OPENBLAS_CFLAGS "openblas_config.h" $INCLUDE_PATHS
    add_lib_from_paths OPENBLAS_LDFLAGS "libopenblas.*" $LIB_PATHS
    ;;
  __DONTUSE__) ;;

  *)
    echo "==================== Linking LAPACK to user paths ===================="
    pkg_install_dir="$with_openblas"
    check_dir "${pkg_install_dir}/include"
    check_dir "${pkg_install_dir}/lib"
    OPENBLAS_CFLAGS="-I'${pkg_install_dir}/include'"
    OPENBLAS_LDFLAGS="-L'${pkg_install_dir}/lib' -Wl,-rpath,'${pkg_install_dir}/lib'"
    OPENBLAS_LIBS="-lopenblas"
    # detect separate omp builds
    (__libdir="${pkg_install_dir}/lib" LIB_PATHS="__libdir" check_lib -lopenblas_openmp 2> /dev/null) &&
      OPENBLAS_LIBS="-lopenblas_openmp"
    (__libdir="${pkg_install_dir}/lib" LIB_PATHS="__libdir" check_lib -lopenblas_omp 2> /dev/null) &&
      OPENBLAS_LIBS="-lopenblas_omp"
    ;;
esac
if [ "$with_openblas" != "__DONTUSE__" ]; then
  if [ "$with_openblas" != "__SYSTEM__" ]; then
    cat << EOF > "${BUILDDIR}/setup_openblas"
prepend_path LD_LIBRARY_PATH "$pkg_install_dir/lib"
prepend_path LD_RUN_PATH "$pkg_install_dir/lib"
prepend_path LIBRARY_PATH "$pkg_install_dir/lib"
prepend_path PKG_CONFIG_PATH "$pkg_install_dir/lib/pkgconfig"
prepend_path CMAKE_PREFIX_PATH "$pkg_install_dir"
prepend_path CPATH "$pkg_install_dir/include"
export LD_LIBRARY_PATH="$pkg_install_dir/lib:"\${LD_LIBRARY_PATH}
export LD_RUN_PATH="$pkg_install_dir/lib:"\${LD_RUN_PATH}
export LIBRARY_PATH="$pkg_install_dir/lib:"\${LIBRARY_PATH}
export CPATH="$pkg_install_dir/include:"\${CPATH}
export PKG_CONFIG_PATH="$pkg_install_dir/lib/pkgconfig:"\${PKG_CONFIG_PATH}
export CMAKE_PREFIX_PATH="$pkg_install_dir:"\${CMAKE_PREFIX_PATH}
export OPENBLAS_ROOT=${pkg_install_dir}
EOF
    cat "${BUILDDIR}/setup_openblas" >> $SETUPFILE
  fi
  cat << EOF >> "${BUILDDIR}/setup_openblas"
export OPENBLAS_ROOT="${pkg_install_dir}"
export OPENBLAS_CFLAGS="${OPENBLAS_CFLAGS}"
export OPENBLAS_LDFLAGS="${OPENBLAS_LDFLAGS}"
export OPENBLAS_LIBS="${OPENBLAS_LIBS}"
export MATH_CFLAGS="\${MATH_CFLAGS} ${OPENBLAS_CFLAGS}"
export MATH_LDFLAGS="\${MATH_LDFLAGS} ${OPENBLAS_LDFLAGS}"
export MATH_LIBS="\${MATH_LIBS} ${OPENBLAS_LIBS}"
export PKG_CONFIG_PATH="${pkg_install_dir}/lib/pkgconfig"
export CMAKE_PREFIX_PATH="${pkg_install_dir}"
prepend_path PKG_CONFIG_PATH "$pkg_install_dir/lib/pkgconfig"
prepend_path CMAKE_PREFIX_PATH "$pkg_install_dir"
EOF
fi

load "${BUILDDIR}/setup_openblas"
write_toolchain_env "${INSTALLDIR}"

cd "${ROOTDIR}"
report_timing "openblas"

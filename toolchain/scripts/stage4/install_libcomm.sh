#!/bin/bash -e

# TODO: Review and if possible fix shellcheck errors.
# shellcheck disable=all
# libcomm is not need any complex setting
# Only problem is the installation from github.com
# LibComm is under highly-active development, the git submodule installation is more recommended

# Last Update in 2025-0504
# other contributor: Peize Lin

[ "${BASH_SOURCE[0]}" ] && SCRIPT_NAME="${BASH_SOURCE[0]}" || SCRIPT_NAME=$0
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_NAME")/.." && pwd -P)"

libcomm_ver="master"
libcomm_sha256="--no-checksum"
source "${SCRIPT_DIR}"/common_vars.sh
source "${SCRIPT_DIR}"/tool_kit.sh
source "${SCRIPT_DIR}"/signal_trap.sh
source "${INSTALLDIR}"/toolchain.conf
source "${INSTALLDIR}"/toolchain.env

[ -f "${BUILDDIR}/setup_libcomm" ] && rm "${BUILDDIR}/setup_libcomm"

libcomm_CFLAGS=""
! [ -d "${BUILDDIR}" ] && mkdir -p "${BUILDDIR}"
cd "${BUILDDIR}"

case "$with_libcomm" in
  __INSTALL__)
    echo "==================== Installing LIBCOMM ===================="
    dirname="LibComm-${libcomm_ver}"
    pkg_install_dir="${INSTALLDIR}/$dirname"
    #pkg_install_dir="${HOME}/lib/libcomm/${libcomm_ver}"
    install_lock_file="$pkg_install_dir/install_successful"
    # url="https://github.com/abacusmodeling/LibComm/archive/refs/tags/v${libcomm_ver}.tar.gz"
    filename="LibComm-${libcomm_ver}.tar.gz"
    url="https://codeload.github.com/abacusmodeling/LibComm/tar.gz/${libcomm_ver}"
    if verify_checksums "${install_lock_file}"; then
        echo "$dirname is already installed, skipping it."
    else
        if [ -f $filename ]; then
        echo "$filename is found"
        else
        # download from github.com and checksum
            echo "===> Notice: This version of LibComm is downloaded in GitHub master repository  <==="
            download_pkg_from_url "${libcomm_sha256}" "${filename}" "${url}"
        fi
    if [ "${PACK_RUN}" = "__TRUE__" ]; then
      echo "--pack-run mode specified, skip installation"
    else
        echo "Installing from scratch into ${pkg_install_dir}"
        [ -d $dirname ] && rm -rf $dirname
        tar -xzf $filename
        # unzip -q $filename
        cp -r $dirname "${pkg_install_dir}/"
        write_checksums "${install_lock_file}" "${SCRIPT_DIR}/stage4/$(basename ${SCRIPT_NAME})"
    fi
    fi
    LIBCOMM_CFLAGS="-I'${pkg_install_dir}'"
        ;;
    __SYSTEM__)
        echo "==================== CANNOT Finding LIBCOMM from system paths NOW ===================="
        recommend_offline_installation $filename $url
        # How to do it in libcomm? -- Zhaoqing in 2023/08/23
        # check_lib -lxcf03 "libxc"
        # check_lib -lxc "libxc"
        # add_include_from_paths LIBXC_CFLAGS "xc.h" $INCLUDE_PATHS
        # add_lib_from_paths LIBXC_LDFLAGS "libxc.*" $LIB_PATHS
        ;;
    __DONTUSE__) ;;
    
    *)
    echo "==================== Linking LIBCOMM to user paths ===================="
    check_dir "${pkg_install_dir}"
    LIBCOMM_CFLAGS="-I'${pkg_install_dir}'"
    ;;
esac
if [ "$with_libcomm" != "__DONTUSE__" ]; then
    if [ "$with_libcomm" != "__SYSTEM__" ]; then
        cat << EOF > "${BUILDDIR}/setup_libcomm"
prepend_path CPATH "$pkg_install_dir/include"
export CPATH="${pkg_install_dir}/include":\${CPATH}
EOF
        cat "${BUILDDIR}/setup_libcomm" >> $SETUPFILE
    fi
    cat << EOF >> "${BUILDDIR}/setup_libcomm"
export LIBCOMM_CFLAGS="${libcomm_CFLAGS}"
export LIBCOMM_ROOT="$pkg_install_dir"
EOF
fi

load "${BUILDDIR}/setup_libcomm"
write_toolchain_env "${INSTALLDIR}"

cd "${ROOTDIR}"
report_timing "libcomm"

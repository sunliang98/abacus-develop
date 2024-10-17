#!/bin/bash -e

# TODO: Review and if possible fix shellcheck errors.
# shellcheck disable=all
# CEREAL is not need any complex setting
# Only problem is the installation from github.com

# Last Update in 2023-0918

[ "${BASH_SOURCE[0]}" ] && SCRIPT_NAME="${BASH_SOURCE[0]}" || SCRIPT_NAME=$0
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_NAME")/.." && pwd -P)"

cereal_ver="1.3.2"
cereal_sha256="16a7ad9b31ba5880dac55d62b5d6f243c3ebc8d46a3514149e56b5e7ea81f85f"
source "${SCRIPT_DIR}"/common_vars.sh
source "${SCRIPT_DIR}"/tool_kit.sh
source "${SCRIPT_DIR}"/signal_trap.sh
source "${INSTALLDIR}"/toolchain.conf
source "${INSTALLDIR}"/toolchain.env

[ -f "${BUILDDIR}/setup_cereal" ] && rm "${BUILDDIR}/setup_cereal"

CEREAL_CFLAGS=""
! [ -d "${BUILDDIR}" ] && mkdir -p "${BUILDDIR}"
cd "${BUILDDIR}"

case "$with_cereal" in
  __INSTALL__)
    echo "==================== Installing CEREAL ===================="
    dirname="cereal-${cereal_ver}"
    pkg_install_dir="${INSTALLDIR}/$dirname"
    #pkg_install_dir="${HOME}/lib/cereal/${cereal_ver}"
    install_lock_file="$pkg_install_dir/install_successful"
    url="https://github.com/USCiLab/cereal/archive/refs/tags/v${cereal_ver}.tar.gz"
    filename="cereal-${cereal_ver}.tar.gz"
    if verify_checksums "${install_lock_file}"; then
        echo "$dirname is already installed, skipping it."
    else
        if [ -f $filename ]; then
        echo "$filename is found"
        else
        # download from github.com and checksum
            echo "===> Notice: This version of CEREAL is downloaded in GitHub Release, which will always be out-of-date version <==="
            download_pkg_from_url "${cereal_sha256}" "${filename}" "${url}"
            # echo "wget $url -O $filename"
            # if ! wget $url -O $filename; then
            # report_error "failed to download $url"
            # recommend_offline_installation $filename $url
            # fi
            # # checksum
            # checksum "$filename" "$cereal_sha256"
        fi
        echo "Installing from scratch into ${pkg_install_dir}"
        [ -d $dirname ] && rm -rf $dirname
        tar -xzf $filename
        mkdir -p "${pkg_install_dir}"
        cp -r $dirname/* "${pkg_install_dir}/"
        write_checksums "${install_lock_file}" "${SCRIPT_DIR}/stage4/$(basename ${SCRIPT_NAME})"
    fi
        ;;
    __SYSTEM__)
        echo "==================== CANNOT Finding CEREAL from system paths NOW ===================="
        recommend_offline_installation $filename $url
        # How to do it in cereal? -- Zhaoqing in 2023/08/23
        # check_lib -lxcf03 "libxc"
        # check_lib -lxc "libxc"
        # add_include_from_paths LIBXC_CFLAGS "xc.h" $INCLUDE_PATHS
        # add_lib_from_paths LIBXC_LDFLAGS "libxc.*" $LIB_PATHS
        ;;
    __DONTUSE__) ;;
    
    *)
    echo "==================== Linking CEREAL to user paths ===================="
    check_dir "${pkg_install_dir}"
    CEREAL_CFLAGS="-I'${pkg_install_dir}'"
    ;;
esac
if [ "$with_cereal" != "__DONTUSE__" ]; then
    if [ "$with_cereal" != "__SYSTEM__" ]; then
    # LibRI deps should find cereal include in CPATH
        cat << EOF > "${BUILDDIR}/setup_cereal"
prepend_path CPATH "$pkg_install_dir/include"
export CPATH="${pkg_install_dir}/include":\${CPATH}
EOF
        cat "${BUILDDIR}/setup_cereal" >> $SETUPFILE
    fi
    cat << EOF >> "${BUILDDIR}/setup_cereal"
export CEREAL_CFLAGS="${CEREAL_CFLAGS}"
export CEREAL_ROOT="$pkg_install_dir"
EOF
fi

load "${BUILDDIR}/setup_cereal"
write_toolchain_env "${INSTALLDIR}"

cd "${ROOTDIR}"
report_timing "cereal"

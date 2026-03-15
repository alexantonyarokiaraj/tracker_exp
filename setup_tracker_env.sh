#!/bin/bash
set -e

source /srv/nfs/soft/spack_latest/spack/share/spack/setup-env.sh

export GCC48=/srv/tf/soft/gcc-4.8.5
export PATH=$GCC48/bin:${PATH}

export SPACK_USER_CONFIG_PATH=$HOME/.spack
export SPACK_ENV_DIR=$HOME/spack_envs

export OPENSSL_HOME=/home2/user/u0100486/linux/openssl102q
export LD_LIBRARY_PATH=$OPENSSL_HOME/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=$OPENSSL_HOME/lib:${LIBRARY_PATH}
export CPATH=$OPENSSL_HOME/include:${CPATH}

export ROOT_INSTALL=/home2/user/u0100486/linux/doctorate/software/root-6.18.04/root_install
source $ROOT_INSTALL/bin/thisroot.sh

export CC=/srv/tf/soft/gcc-4.8.5/bin/gcc
export CXX=/srv/tf/soft/gcc-4.8.5/bin/g++

export X11_HOME=/home2/user/u0100486/spack_envs/root-gcc48/.spack-env/view
export CMAKE_INCLUDE_PATH=$X11_HOME/include:${CMAKE_INCLUDE_PATH}
export CMAKE_LIBRARY_PATH=$X11_HOME/lib:${CMAKE_LIBRARY_PATH}

export TBB_ROOT=$HOME/tbb
export LD_LIBRARY_PATH=$TBB_ROOT/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=$TBB_ROOT/lib:${LIBRARY_PATH}
export CPATH=$TBB_ROOT/include:${CPATH}

spack env activate ~/spack_envs/root-gcc48/

source /home2/user/u0100486/linux/doctorate/packages/geant4.10.5-install/bin/geant4.sh
source /home2/user/u0100486/linux/doctorate/packages/geant4.10.5-install/share/Geant4-10.5.1/geant4make/geant4make.sh
source /home2/user/u0100486/linux/doctorate/github/iTracks/simulations/nptool/nptool.sh

export PATH=$HOME/.local/bin:${PATH}

source ~/my_python_env/bin/activate

export LD_LIBRARY_PATH=$HOME/libs/lerc/lib:${LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1

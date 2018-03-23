#!/bin/bash

cd "$(dirname "$0")"

virtualenv build_env
source build_env/bin/activate
pybuilderInstalled=`pip freeze | grep 'pybuilder' | wc -l`

if [ $pybuilderInstalled != 1 ]
then
   echo "Installing pybuilder"
   pip install pybuilder
fi

pyb install_dependencies clean publish

cp -TRv target/dist/mlworkbench-0.0.1/dist/ ../install/
rm -rf build_env
rm -rf target

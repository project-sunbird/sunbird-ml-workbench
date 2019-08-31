#!/bin/bash


cd "$(dirname "$0")"

pybuilderInstalled=`pip freeze | grep 'pybuilder' | wc -l`

if [ $pybuilderInstalled != 1 ]
then
   echo "Installing pybuilder"
   pip install pybuilder
fi

pyb install_dependencies clean publish
pyb sphinx_generate_documentation
tox

rm -rf mlwb_venv/

if [ ! -d "bin" ]; then
  mkdir 'bin'
fi

cp target/dist/daggit-0.5.0/dist/* bin/

#rm -rf target/

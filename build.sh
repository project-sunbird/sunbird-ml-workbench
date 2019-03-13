#!/bin/bash
pip install pybuilder
pyb install_dependencies clean publish
pyb sphinx_generate_documentation
tox
mkdir -p bin/
cp target/dist/daggit-0.5.0/dist/* bin/

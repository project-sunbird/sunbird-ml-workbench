#!/bin/bash

pip install pybuilder
pyb install_dependencies clean publish
pyb sphinx_generate_documentation
tox

# Extracting the Python installer for daggit from the above built source
mkdir -p bin/
cp target/dist/daggit-0.5.0/dist/* bin/

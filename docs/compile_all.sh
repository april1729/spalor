#! /bin/sh

jupyter nbconvert source/user_guide/*.ipynb --to rst
jupyter nbconvert ../examples/*.ipynb --to rst
cp ../examples/* source/examples/


sphinx-build source/ .

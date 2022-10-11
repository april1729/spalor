#! /bin/sh

rm *.html
rm -r _*
rm -r api_doc
rm -r user_guide
rm -r examples

jupyter nbconvert source/user_guide/*.ipynb --to rst
jupyter nbconvert ../examples/*.ipynb --to rst
cp -r ../examples/* source/examples/


sphinx-build source/ .

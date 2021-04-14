import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='SpaLoR',
     version='0.1',
     scripts=[] ,
     author="April Sagan",
     author_email="aprilsagan1729@gmail.com",
     description="A python library for sparse and low rank matrix methods used in machine learning, like Matrix Completion, Robust PCA, and CUR factorizations",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="www.spalor.org",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU Affero General Public License v3",
         "Operating System :: OS Independent",
         "Topic :: Scientific/Engineering :: Mathematics"
     ],
 )

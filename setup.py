from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='calsagos',
      version='0.1',
      description='CALSAGOS: Clustering ALgorithmS Applied to Galaxies in Overdense Systems',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Daniela Olave-Rojas & Pierluigi Cerulo',
      author_email='daniela.olave@utalca.cl - pcerulo@inf.udec.cl',
      url = "https://github.com/dolaver/calsagos",
      packages = ['calsagos'],  #same as name
      )

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="calsagos",
    version="1.0",
    description="CALSAGOS: Clustering ALgorithmS Applied to Galaxies in Overdense Systems",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniela Olave-Rojas & Pierluigi Cerulo",
    author_email="daniela.olave@utalca.cl - pcerulo@inf.udec.cl",
    url = 'https://github.com/dolaver/calsagos',
    download_url = 'https://github.com/dolaver/calsagos/1.0',
    packages=['calsagos'],  #same as name
    keywords = ['astronomy', 'utilities', 'substructures'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": ""},
    packages=setuptools.find_packages(where=""),
    python_requires=">=3.6",
)


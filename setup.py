import setuptools
from setuptools import setup

setup(
    name="porteratzo3D",
    version="0.1.0",
    description="Collection of utils",
    url="https://github.com/porteratzo/porteratzo3D.git",
    author="Omar Montoya",
    author_email="omar.alfonso.montoya@hotmail.com",
    license="MIT License",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "open3d",
        "porter_bench @ git+https://github.com/porteratzo/tictoc.git@main",
    ],
    python_requires='>=3.7,<3.11',
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
    ],
)

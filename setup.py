#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="manifold-gp",
    version="1.1.0",
    author="Bernardo Fichera",
    author_email="bernardo.fichera@gmail.com",
    description="Manifold Informed Gaussian Process.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nash169/manifold-gp.git",
    packages=find_namespace_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",                # math
        "matplotlib",           # plotting
        "torch",                # net framework
        "torch-scatter",
        "torch-sparse",
        "faiss",
        "gpytorch"
    ],
    extras_require={
        "pytorch": [
            "torchvision",      # net framework GPU
            "torchaudio",
            "tensorflow"
        ],
        "dev": [
            "pylint",           # python linter
        ]
    },
    package_data={
        "manifold_gp.data": ["*.msh", "*.stl", "*.csv"],
    }
)

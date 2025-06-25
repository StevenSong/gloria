import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="gloria",
    py_modules=["gloria"],
    version="0.1",
    description="",
    author="Shih-Cheng Huang (Mars)",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[],
    include_package_data=True,
    license="Apache License",
)

#!/usr/bin/env python

from setuptools import find_packages, setup

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read requirements_dev.txt
with open("requirements_dev.txt") as f:
    dev_requirements = f.read().splitlines()[1:]

setup(
    name="tali",
    version="2.1.0",
    description="TALI: A quadra model dataset and transforms for PyTorch",
    author="Antreas Antoniou",
    author_email="iam@antreas.io",
    install_requires=requirements,  # List of requirements loaded from requirements.txt
    extras_require={
        "dev": dev_requirements  # List of dev requirements loaded from requirements_dev.txt
    },
    packages=find_packages(),
)

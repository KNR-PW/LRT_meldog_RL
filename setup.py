# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Setup script for meldog_rl package."""

from setuptools import setup, find_packages

setup(
    name="meldog_rl",
    version="0.1.0",
    description="Quadruped robot locomotion with reinforcement learning",
    author="Meldog Team",
    python_requires=">=3.10",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    install_requires=[
        # Isaac Lab dependencies are managed separately
        "torch>=2.0",
        "numpy",
        "h5py",
        "prettytable",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
    },
    entry_points={
        "console_scripts": [
            # Could add CLI entry points here
        ],
    },
)

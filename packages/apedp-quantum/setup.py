from setuptools import setup, find_packages

setup(
    name="apedp-quantum",
    version="0.1.0",
    description="Autonomous Quantum Many-Body Discovery, DMRG & Tensor-Network Platform",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
    ],
)

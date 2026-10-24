from setuptools import setup, find_packages

setup(
    name="apedp-molecular",
    version="0.1.0",
    description="Autonomous Molecular SE(3)-Equivariant Flow & Energy Relaxation Platform",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "apedp>=1.0.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
    ],
)

from setuptools import setup, find_packages

setup(
    name="apedp-continuum",
    version="0.1.0",
    description="Autonomous High-Dimensional Continuous PDE & Coupled ODE Discovery Platform",
    author="APEDP-X Discovery Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "pandas>=1.4.0",
    ],
    extras_require={
        "neural": [
            "torch>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)

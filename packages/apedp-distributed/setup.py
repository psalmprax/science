from setuptools import setup, find_packages

setup(
    name="apedp-distributed",
    version="0.1.0",
    description="Autonomous Distributed & Out-of-Core Chunked Sweep Execution Framework",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "apedp>=1.0.0",
    ],
    extras_require={
        "ray": ["ray>=2.0.0"],
    },
)

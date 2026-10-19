from setuptools import setup, find_packages

setup(
    name="apedp-robotics",
    version="0.1.0",
    description="Autonomous Robotics Trajectory Transport, SE(2)/SE(3) Kinematics & Control",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "apedp>=1.0.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
    ],
)

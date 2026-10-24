from setuptools import setup, find_packages

setup(
    name="apedp-audio",
    version="0.1.0",
    description="Autonomous Audio Spectrogram Field Transport, STFT Ridge Tracking & Jet Advection",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "apedp>=1.0.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
    ],
)

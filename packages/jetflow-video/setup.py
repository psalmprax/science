from setuptools import setup, find_packages

setup(
    name="jetflow-video",
    version="0.1.0",
    description="Continuous High-Dimensional Spatio-Temporal PDE Video Generation & 4D Tensor-Train Compression",
    author="APEDP-X Discovery Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "torch>=2.0.0",
        "pillow>=9.0.0",
        "imageio[ffmpeg]>=2.20.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "diffusers": [
            "diffusers>=0.20.0",
            "transformers>=4.25.0",
            "accelerate>=0.20.0",
        ],
    },
)

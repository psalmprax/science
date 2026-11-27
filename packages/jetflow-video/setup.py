"""Build script of the standalone ``jetflow-video`` wheel.

The wheel is standalone by contract: nothing it ships may import ``apedp`` or ``apedp_dyn``
(``tests/test_dyn_all_subpackages.py`` and the wheel workflow check it). One module of this source
tree is therefore **not shipped**: ``jetflow_video/head_to_head.py``, the comparison harness of two
preregistered runs of the APEDP-X repository. It is frozen by those runs' digests, so it cannot be
moved out of this directory, and one of its functions imports that repository's toy worlds. The
runs read it from the source tree; the wheel leaves it out.
"""
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

#: ``(package, module)`` pairs that stay in the source tree and are left out of the wheel.
NOT_SHIPPED = {("jetflow_video", "head_to_head")}


class build_py(_build_py):
    """``build_py`` without the modules named in ``NOT_SHIPPED``."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [m for m in modules if (m[0], m[1]) not in NOT_SHIPPED]


setup(
    cmdclass={"build_py": build_py},
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

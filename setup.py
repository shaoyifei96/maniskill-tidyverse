from setuptools import setup, find_packages
import os

# setup.py lives inside maniskill_tidyverse/ itself.
# We install the parent directory as package "maniskill_tidyverse"
# by pointing package_dir at "." → maniskill_tidyverse.
setup(
    name="maniskill-tidyverse",
    version="0.2.0",
    package_dir={"maniskill_tidyverse": "."},
    packages=["maniskill_tidyverse"] + [
        f"maniskill_tidyverse.{p}"
        for p in find_packages(where=".")
    ],
    package_data={"": ["*.urdf", "*.srdf", "*.glb", "*.dae", "*.stl"]},
    include_package_data=True,
    install_requires=[
        "mani_skill",
        "torch",
        "numpy",
        "mplib>=0.2.0",
        "transforms3d",
        "opencv-python",
        "pytorch_kinematics",
    ],
)

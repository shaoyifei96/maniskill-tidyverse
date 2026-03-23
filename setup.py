from setuptools import setup, find_packages

setup(
    name="maniskill-tidyverse",
    version="0.1.0",
    py_modules=["tidyverse_agent"],
    packages=find_packages(),
    package_data={"": ["*.urdf", "*.srdf", "*.glb", "*.dae", "*.stl"]},
    include_package_data=True,
    install_requires=["mani_skill"],
)

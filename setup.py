from setuptools import find_packages, setup

setup(
    name="vidfm3d",
    version="1.0",
    description="Spatial diagnostic probes for frozen video foundation models",
    python_requires=">=3.11",
    author="Zixuan Huang",
    author_email="zixuan32@illinois.edu",
    packages=find_packages(include=["vidfm3d", "vidfm3d.*"]),
)

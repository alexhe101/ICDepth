import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="icdepth",
    version="0.1.0",
    description="ICDepth: Taming Video Diffusion Models for Video Depth Estimation via In-Context Conditioning",
    url="https://github.com/alexhe101/ICDepth",
    license="Apache-2.0",
    packages=find_packages(include=["icdepth", "icdepth.*"]),
    install_requires=install_requires,
    python_requires=">=3.9",
)

from setuptools import find_packages, setup
import os
import sys

path = os.path.dirname(__file__)
os.chdir(path)
with open("./python/jnerf/__init__.py", "r") as f:
    src = f.read()
version = src.split("__version__")[1].split("'")[1]
print("setup jnerf version", version)

setup(
    name="jnerf",
    version=version,
    description="NeRF benchmark based on Jittor",
    author="jnerf",
    url="https://github.com/Jittor/JNeRF",
    packages=["jnerf"],
    package_dir={'': 'python'},
    package_data={'': [ "*"+"/*"*i for i in range(20)]},
    install_requires=[
        "jittor>=1.3.5.25",
        "numpy",
        "tqdm",
        "opencv-python",
        "Pillow",
        "imageio",
        "pyyaml",
        "PyMCubes",
        "trimesh",
        "open3d"
    ]
)


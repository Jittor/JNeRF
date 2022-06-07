from setuptools import find_packages, setup

setup(
    name="jnerf",
    version="0.1.0",
    description="NeRF benchmark based on Jittor",
    author="jnerf",
    url="https://github.com/Jittor/JNeRF",
    packages=find_packages(),
    install_requires=[
        "jittor>=1.3.4.13",
        "numpy",
        "tqdm",
        "opencv-python",
        "Pillow",
        "imageio",
        "pyyaml"
    ]
)


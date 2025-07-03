from setuptools import setup, find_packages

setup(
    name="mlx",                   # <— so pip install mlx will use your local code
    version="0.1.0",
    packages=find_packages(),     # make sure your `mlx/` folder has an __init__.py
    install_requires=["mpi4py"],
)

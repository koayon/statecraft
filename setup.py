from setuptools import find_packages, setup

setup(
    name="statecraft",
    version="0.1",
    packages=find_packages(exclude=["tests", "docs"]),
)

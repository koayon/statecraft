from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="statecraft",
    version="0.1.1",
    packages=find_packages(exclude=["tests", "docs"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
)

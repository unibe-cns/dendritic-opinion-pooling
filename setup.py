# encoding: utf8
import re

from setuptools import setup


def _cut_version_number_from_requirement(req):
    return req.split()[0]


def read_metadata(metadata_str):
    """
    Find __"meta"__ in init file.
    """
    with open("./dopp/__init__.py", "r") as f:
        meta_match = re.search(fr"^__{metadata_str}__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError(f"Unable to find __{metadata_str}__ string.")


def read_requirements():
    requirements = []
    with open("./requirements.txt") as f:
        for req in f:
            req = req.replace("\n", " ")
            requirements.append(req)
    return requirements

def read_long_description():
    with open("README.md", "r") as f:
        descr = f.read()
    return descr


setup(
    name="dopp",
    version=read_metadata("version"),
    maintainer=read_metadata("maintainer"),
    author=read_metadata("author"),
    description=(read_metadata("description")),
    license=read_metadata("license"),
    keywords=("simulation", "neuronal networks", "dendritic computation", "synaptic plasticity"),
    url=read_metadata("url"),
    python_requires=">=3.6, <4",
    install_requires=read_requirements(),
    packages=["dopp"],
    long_description=read_long_description(),
    long_description_content_type="text/x-rst",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
)

import codecs
from setuptools import setup, find_packages

with codecs.open("README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="prospr",
    description="Prospect Pruning: Finding Trainable Weights at Initialization Using Meta-Gradients",
    long_description=README,
    long_description_content_type="text/markdown",
    version="1.0",
    packages=find_packages(),
    url="https://github.com/mil-ad/prospr",
    author="Milad Alizadeh",
    author_email="milad.alizadeh@cs.ox.ac.uk",
    install_requires=["torch>=1.9.1"],
    python_requires=">=3.8.5",
)

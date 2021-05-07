"""Installation script for easy pip install."""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="octopytorch",
    version="0.3",
    description="Segmentation Networks without a Backbone",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicolas Pielawski",
    author_email="nicolas@pielawski.fr",
    packages=setuptools.find_packages(),
    install_requires=["torch>=1.4"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

from setuptools import setup, find_packages

# get version number
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="First_package_Rosa",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy"
    ],
    author="Jan Rosa",
    url="https://github.com/janrosa1/my_package",
    author_email="jan.rosa1993@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/janrosa1/my_package",
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        ],
)

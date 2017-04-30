import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "sta_663",
    version = "1.0.1",
    author = "Bohao Huang, Siyang Yuan",
    author_email = "bohao.huang@duke.edu, siyang.yuan@duke.edu",
    description = ("A class project of STA663, implementation of Variational Autoencoder"),
    license = "MIT",
    keywords = "Variational Autoencoder",
    url = "https://github.com/qingshuimonk/STA663.git",
    packages=['vae'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
    install_requires=[
        'numpy',
    ],
)
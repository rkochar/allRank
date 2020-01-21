# coding=utf-8

from setuptools import setup, find_packages

reqs = [
    "tensorflow==1.13.1",
    "torch==1.2.0",
    "torchvision==0.4.0",
    "scikit-learn==0.21.3",
    "pandas==0.24.2",
    "numpy==1.16.1",
    "scipy>=1.2.0",
    "attrs==18.2.0",
    "flatten_dict==0.1.0",
    "pillow==6.2.1"
]

setup(
    name="allRank",
    version="1.1.0",
    license="Apache 2",
    url="https://github.com/allegro/allRank",
    install_requires=reqs,
    author_email="allrank@allegro.pl",
    description="allRank is a framework for training learning-to-rank neural models",
    packages=find_packages(exclude=["tests"]),
    package_data={"allrank": ["config.json"]},
    entry_points={"console_scripts": ['allRank = allrank.main:run']},
    zip_safe=False,
)

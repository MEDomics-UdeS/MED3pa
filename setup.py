import os
from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name="MED3pa",
    version="0.1.15",
    author="MEDomics consortium",
    author_email="medomics.info@gmail.com",
    description="Python Open-source package for ensuring robust and reliable ML models deployments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyna1404/MED3pa",
    project_urls={
        'Documentation': 'https://med3pa.readthedocs.io/en/latest/',
        'Github': 'https://github.com/lyna1404/MED3pa'
    },
    packages=find_packages(exclude=['docs', 'tests', 'experiments']),
    python_requires='>=3.9',
    install_requires=requirements,
)

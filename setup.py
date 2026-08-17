import os
from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name="MED3pa",
    version="1.1.0a1",
    author="MEDomics consortium",
    author_email="medomics.info@gmail.com",
    description="Python Open-source package for ensuring robust and reliable ML models deployments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MEDomicsLab/MED3pa",
    project_urls={
        'Documentation': 'https://med3pa.readthedocs.io/en/latest/',
        'Github': 'https://github.com/MEDomicsLab/MED3pa'
    },
    packages=find_packages(exclude=['docs', 'tests', 'experiments']),
include_package_data=True,
    package_data={
            "MED3pa": ["MED3pa/visualization/tree_template/*"]
        },
    python_requires='>=3.9',
    install_requires=requirements,
)

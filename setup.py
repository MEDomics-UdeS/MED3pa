import sys

from setuptools import find_packages, setup

# Check if current python installation is >= 3.9
if sys.version_info < (3, 9, 0):
  raise Exception("MED3pa requires python 3.9 or later")

with open("README.md", encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name="MED3pa",
    version="0.1.0",
    author="MEDomics consortium",
    author_email="medomics.info@gmail.com",
    description="Python Open-source package for ensuring robust and reliable ML models deployments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyna1404/MED3pa",
    project_urls={
        'Github': 'https://github.com/lyna1404/MED3pa'
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='machine learning, covariate shift, uncertainty, robust models, problematic profiles, AI',
    python_requires='>=3.9,<=3.12.3',
    packages=find_packages(exclude=['docs', 'tests', 'experiments']),
    install_requires=requirements
)
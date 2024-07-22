# MED3pa Package

## Table of Contents
- [Overview](#overview)
- [Key Functionalities](#key-functionalities)
- [Subpackages](#subpackages)
- [Getting Started with the Package](#getting-started)
    - [Installation](#installation)
    - [A Simple Example](#a-simple-example)
- [Tutorials](#tutorials)
- [Acknowledgement](#acknowledgement)
- [References](#references)
- [Authors](#authors)
- [Statement](#statement)
- [Supported Python Versions](#supported-python-versions)

## Overview

<img src="https://github.com/lyna1404/MED3pa/blob/main/docs/diagrams/package_white_bg.svg" alt="Overview" style="width:100%;">

The **MED3pa** package is specifically designed to address critical challenges in deploying machine learning models, particularly focusing on the robustness and reliability of models under real-world conditions. It provides comprehensive tools for evaluating model stability and performance in the face of **covariate shifts**, **uncertainty**, and **problematic data profiles**.

## Key Functionalities

- **Covariate Shift Detection**: Utilizing the Detectron subpackage, MED3pa can identify significant shifts in data distributions that might affect the model’s predictions. This feature is crucial for applications such as healthcare, where early detection of shifts can prevent erroneous decisions.

- **Uncertainty and Confidence Estimation**: Through the med3pa subpackage, the package measures the uncertainty and predictive confidence at both individual and group levels. This helps in understanding the reliability of model predictions and in making informed decisions based on model outputs.

- **Identification of Problematic Profiles**: MED3pa analyzes data profiles that consistently lead to poor model performance. This capability allows developers to refine training datasets or retrain models to handle these edge cases effectively.

## Subpackages

<p align="center">
    <img src="https://github.com/lyna1404/MED3pa/blob/main/docs/diagrams/subpackages.svg" alt="Overview">
</p>

The package is structured into four distinct subpackages:

- **datasets**: Stores and manages the dataset.
- **models**: Handles ML models operations.
- **detectron**: Evaluates the model against covariate shift.
- **med3pa**: Evaluates the model’s performance & extracts problematic profiles.

This modularity allows users to easily integrate and utilize specific functionalities tailored to their needs without dealing with unnecessary complexities.

## Getting Started with the Package

To get started with MED3pa, follow the installation instructions and usage examples provided in the documentation.

### Installation

```bash
pip install MED3pa
```

### A simple exemple

```python
   
    from MED3pa.datasets import DatasetsManager
    from MED3pa.models import BaseModelManager, ModelFactory
    from MED3pa.med3pa import Med3paDetectronExperiment
    
    # Initialize the DatasetsManager
    datasets = DatasetsManager()

    # Load datasets for training, validation, reference, and testing
    datasets.set_from_file(dataset_type="training", file='./tutorials/data/train_data.csv', target_column_name='Outcome')
    datasets.set_from_file(dataset_type="validation", file='./tutorials/data/val_data.csv', target_column_name='Outcome')
    datasets.set_from_file(dataset_type="reference", file='./tutorials/data/test_data.csv', target_column_name='Outcome')
    datasets.set_from_file(dataset_type="testing", file='./tutorials/data/test_data_shifted_0.6.csv', target_column_name='Outcome')


    # Initialize the model factory and load the pre-trained model
    factory = ModelFactory()
    model = factory.create_model_from_pickled("./tutorials/models/diabetes_xgb_model.pkl")

    # Set the base model using BaseModelManager
    base_model_manager = BaseModelManager()
    base_model_manager.set_base_model(model=model)

    # Execute the integrated MED3PA and Detectron experiment
    reference_results, test_results, detectron_results = Med3paDetectronExperiment.run(
        datasets=datasets,
        base_model_manager=base_model_manager,
    )

    # Save the results to a specified directory
    reference_det_results.save(file_path='./tutorials/med3pa_detectron_experiment_results/reference')
    test_det_results.save(file_path='./tutorials/med3pa_detectron_experiment_results/test')
    detectron_results.save(file_path='./tutorials/med3pa_detectron_experiment_results/detectron')

```

### Tutorials

We have created many [tutorial notebooks](https://github.com/lyna1404/MED3pa/tree/main/tutorials) to assist you in learning how to use the different parts of the package.


## Acknowledgement
MED3pa is an open-source package developed at the [MEDomics-Udes](https://www.medomics-udes.org/en/) laboratory with the collaboration of the international consortium [MEDomics](https://www.medomics.ai/). We welcome any contribution and feedback. 

## References
This package utilizes the methods described in the following work:

Ginsberg, T., Liang, Z., & Krishnan, R. G. (2023). [A Learning Based Hypothesis Test for Harmful Covariate Shift](https://openreview.net/forum?id=rdfgqiwz7lZ). In *The Eleventh International Conference on Learning Representations*.

## Authors
* [Lyna Chikouche: ](https://www.linkedin.com/in/lynahiba-chikouche-62a5181bb/) Research intern at MEDomics-Udes laboratory.
* [Ludmila Amriou: ](https://www.linkedin.com/in/ludmila-amriou-875b58238//) Research intern at MEDomics-Udes laboratory.
* [Olivier Lefebvre: ](https://www.linkedin.com/in/olivier-lefebvre-bb8837162/) Student (Ph. D. Computer science) at université de Sherbrooke
* [Martin Vallières: ](https://www.linkedin.com/in/martvallieres/) Assistant professor, computer science department at université de Sherbrooke

## Statement

This package is part of https://github.com/medomics, a package providing research utility tools for developing precision medicine applications.

```
Copyright (C) 2024 MEDomics consortium

GPL3 LICENSE SYNOPSIS

Here's what the license entails:

1. Anyone can copy, modify and distribute this software.
2. You have to include the license and copyright notice with each and every distribution.
3. You can use this software privately.
4. You can use this software for commercial purposes.
5. If you dare build your business solely from this code, you risk open-sourcing the whole code base.
6. If you modify it, you have to indicate changes made to the code.
7. Any modifications of this code base MUST be distributed with the same license, GPLv3.
8. This software is provided without warranty.
9. The software author or license can not be held liable for any damages inflicted by the software.
```

More information on about the [LICENSE can be found here](https://github.com/MEDomics-UdeS/MEDimage/blob/main/LICENSE.md)

## Supported Python Versions

The **MED3pa** package is developed and tested with Python 3.12.3.

Additionally, it is compatible with the following Python versions:
- Python 3.11.x
- Python 3.10.x
- Python 3.9.x

While the package may work with other versions of Python, these are the versions we officially support and recommend.
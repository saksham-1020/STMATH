---
title: "STMATH: Unified Math & AI Toolkit for Python"
authors:
  - name: Saksham Tomar
    affiliation: 1
    orcid: "0009-0001-1691-9981"
affiliations:
  - name: Medi-Caps University, Indore, India
    index: 1
date: 2025-12-02
bibliography: references.bib
tags:
  - mathematics
  - scientific-computing
  - machine-learning
  - quantum-computing
  - cryptography
  - python
---
# STMATH: Unified Math & AI Toolkit for Python

## Summary

STMATH is an open-source Python library that provides a unified collection of mathematical, statistical, machine learning, cryptographic, quantum computing, vision, optimization, and general scientific utilities under a single lightweight API. Unlike traditional scientific stacks that require multiple specialized dependencies, STMATH focuses on educational clarity, modularity, and ease of use for students, researchers, and developers.

The library is designed for fast prototyping, classroom instruction, and interdisciplinary research, enabling users to access core mathematical and AI-related functions without complex setup or heavy dependency management.

---

## Statement of Need

Scientific computing in Python is dominated by large, specialized libraries such as NumPy, SciPy, and scikit-learn. While powerful, these ecosystems often impose steep learning curves and heavy dependency structures for beginners and early-stage researchers. Additionally, learners frequently need to combine tools from multiple domains such as statistics, optimization, cryptography, and machine learning, leading to fragmented workflows.

STMATH addresses this problem by offering a unified, lightweight, and modular toolkit that consolidates essential mathematical and AI-related utilities into a single coherent library. It reduces setup friction for educational and experimental workflows while maintaining conceptual clarity and reproducibility. The software is particularly suited for undergraduate education, self-learners, and rapid research prototyping.

---

## Installation

Install STMATH directly from PyPI:

```bash
pip install stmath
 ```
### Upgrade to the latest version:
```
pip install --upgrade stmath
```
---

## Usage Examples

### Basic Mathematical Operations

```python
import stmath as am

print(am.add(10, 5))     # 15
print(am.sqrt(16))      # 4.0
print(am.power(2, 3))   # 8
```
### Statistical Analysis
```python
import stmath as am

data = [1, 2, 3, 4, 5]
print(am.mean(data))       # 3.0
print(am.variance(data))  # 2.5
```
### Machine Learning Metric
```python
import stmath as am

y_true = [1, 0, 1]
y_pred = [1, 0, 0]

print(am.f1_score(y_true, y_pred))   # 0.667
print(am.accuracy(y_true, y_pred))   # 0.667
```
---
### Acknowledgements

The author would like to acknowledge Medi-Caps University, Indore, for academic support. This work did not receive any specific grant from public, commercial, or not-for-profit funding agencies.

---
### References








 



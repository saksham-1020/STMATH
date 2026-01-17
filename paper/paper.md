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
STMATH is an open-source Python library designed primarily for educational use in mathematics, data science, and introductory artificial intelligence. The library provides a unified and lightweight interface for performing common mathematical, statistical, and machine learning operations, making it suitable for undergraduate students, self-learners, and instructors.

Unlike traditional scientific Python stacks that require learners to navigate multiple complex libraries, STMATH emphasizes conceptual clarity and ease of use. It enables students to experiment with mathematical concepts and basic AI metrics using a single, consistent API. The software is intended for classroom demonstrations, lab assignments, and early-stage experimentation in interdisciplinary courses. The STMATH software is archived on Zenodo and cited as an open-source educational resource [@stmath2025].


## Statement of Need

Students learning scientific computing and introductory machine learning are often required to work with multiple specialized Python libraries such as NumPy [@numpy], SciPy [@scipy], and scikit-learn [@scikit]. While these tools are powerful, their combined complexity can be overwhelming for beginners and can distract from core learning objectives.

STMATH addresses this challenge by offering a unified and simplified toolkit that brings together essential mathematical, statistical, and basic machine learning utilities under a single interface. By reducing setup complexity and minimizing cognitive overhead, STMATH allows learners to focus on understanding fundamental concepts rather than managing multiple dependencies. The software is particularly useful in undergraduate coursework, self-guided learning, and rapid prototyping environments. Foundational concepts in machine learning are discussed in standard educational resources such as Goodfellow et al. [@goodfellow2016deep].


## Educational Use and Learning Outcomes

STMATH is designed to support teaching and learning in undergraduate-level mathematics, data science, and introductory artificial intelligence courses. The software can be used by students to explore mathematical operations, statistical measures, and evaluation metrics in a hands-on manner.

By using STMATH, learners can:
- Understand basic mathematical and statistical concepts through simple function calls
- Experiment with machine learning evaluation metrics without requiring full model implementations
- Develop confidence in Python-based scientific computing through a unified and consistent interface

Instructors can integrate STMATH into laboratory sessions, assignments, or demonstrations to illustrate foundational concepts in applied mathematics and AI with minimal setup overhead.

The software is most suitable for first- and second-year undergraduate courses.

## Installation

Install STMATH directly from PyPI:

```bash
pip install stmath
 ```
### Upgrade to the latest version:
```
pip install --upgrade stmath
```

## Usage Examples
The following examples illustrate how STMATH can be used by students to explore core mathematical and machine learning concepts.

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
### Cryptography 
```python
import stmath as am

print(am.sha256("hello"))  
# → "2cf24dba5fb0a30e26e83b2ac5b9e29e"

print(am.gas_fee(gas_used=21000, gwei=50, eth_price=2000))  
# → 2.1 USD (approx)
```
### Quantum Function
```python
import stmath as am

print(am.hadamard([1,0]))   # → [0.707, 0.707]
print(am.pauli_x([1,0]))    # → [0,1]
print(am.pauli_z([1,0]))    # → [1,0]
```

## Acknowledgements

The author would like to acknowledge Medi-Caps University, Indore, for academic support. This work did not receive any specific grant from public, commercial, or not-for-profit funding agencies.

## References









 











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
  - python
---
# STMATH: Unified Math & AI Toolkit for Python

## Summary
STMATH is an open-source Python library designed primarily for educational use in mathematics, data science, and introductory artificial intelligence. The library provides a unified and lightweight interface for performing common mathematical, statistical, and machine learning operations, making it suitable for undergraduate students, self-learners, and instructors.

Unlike traditional scientific Python stacks that require learners to navigate multiple complex libraries, STMATH emphasizes conceptual clarity and ease of use. It enables students to experiment with mathematical concepts and basic AI metrics using a single, consistent API. The software is intended for classroom demonstrations, lab assignments, and early-stage experimentation in interdisciplinary courses. The STMATH software is archived on Zenodo and cited as an open-source educational resource [@stmath2025].

The project is intended to function as an educational bridge between beginner programming experiences and the broader scientific Python ecosystem.


## Statement of Need

Students learning scientific computing and introductory machine learning are often required to work with multiple specialized Python libraries such as NumPy [@numpy], SciPy [@scipy], and scikit-learn [@scikit]. While these tools are powerful, their combined complexity can be overwhelming for beginners and can distract from core learning objectives.

STMATH addresses this challenge by offering a unified and simplified toolkit that brings together essential mathematical, statistical, and basic machine learning utilities under a single interface. By reducing setup complexity and minimizing cognitive overhead, STMATH allows learners to focus on understanding fundamental concepts rather than managing multiple dependencies. The software is particularly useful in undergraduate coursework, self-guided learning, and rapid prototyping environments. Foundational concepts in machine learning are discussed in standard educational resources such as Goodfellow et al. [@goodfellow2016deep].

## State of the Field

Educational computing environments often rely on multiple scientific Python libraries such as NumPy, SciPy, and scikit-learn. While these ecosystems are powerful and widely adopted, beginners frequently encounter challenges due to fragmented APIs, installation complexity, and steep learning curves during early STEM education.

STMATH is positioned as an educational abstraction layer that simplifies introductory experimentation and classroom-oriented workflows through a unified interface. The project is intended to complement—not replace—the broader scientific Python ecosystem.


## Software Design

STMATH follows a modular and lightweight software design focused on educational usability. The toolkit groups commonly used mathematical, statistical, and introductory AI utilities into a consistent interface intended for rapid experimentation and conceptual learning.

The software architecture prioritizes readability, simplified APIs, and ease of integration within beginner-level educational workflows. The design philosophy prioritizes educational accessibility and conceptual learning over high-performance computational optimization.


## Educational Applications and Learning Outcomes

STMATH is designed to support teaching and learning in undergraduate-level mathematics, data science, and introductory artificial intelligence courses. The software can be used by students to explore mathematical operations, statistical measures, and evaluation metrics in a hands-on manner.

By using STMATH, learners can:
- Understand basic mathematical and statistical concepts through simple function calls
- Experiment with machine learning evaluation metrics without requiring full model implementations
- Develop confidence in Python-based scientific computing through a unified and consistent interface

Instructors can integrate STMATH into laboratory sessions, assignments, or demonstrations to illustrate foundational concepts in applied mathematics and AI with minimal setup overhead.

The software is most suitable for first- and second-year undergraduate courses.


## Educational Positioning

STMATH is intended to complement existing scientific Python ecosystems by providing a simplified educational abstraction layer for beginners. The toolkit does not aim to replace mature libraries such as NumPy, SciPy, or scikit-learn, but instead focuses on reducing the initial learning barrier commonly faced by undergraduate STEM learners.

The project emphasizes conceptual clarity, unified workflows, and simplified experimentation so that students can focus on understanding foundational concepts before transitioning to more advanced scientific Python ecosystems.

| Educational Aspect | Traditional Scientific Libraries | STMATH |
|---|---|---|
| Beginner-oriented simplified interface | Partial | Yes |
| Unified introductory workflows | Limited | Yes |
| Minimal setup educational examples | Partial | Yes |
| Classroom-focused experimentation | Limited | Yes |
| Consistent beginner-friendly API style | Partial | Yes |


## Example Learning Workflow

A typical introductory classroom workflow using STMATH may involve:

1. Introducing students to basic mathematical operations and descriptive statistics
2. Demonstrating simplified machine learning evaluation metrics
3. Allowing learners to experiment with unified scientific workflows using minimal setup
4. Transitioning students toward broader scientific Python ecosystems such as NumPy and scikit-learn after foundational understanding is established

This progression supports scaffolded learning and reduces the cognitive overhead often associated with fragmented beginner scientific computing environments.


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

The following examples are intentionally simplified to support conceptual understanding and beginner-level classroom experimentation.

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

### Classroom Statistics Workflow

```python
import stmath as am

scores = [72, 81, 90, 65, 88]

mean_score = am.mean(scores)
variance_score = am.variance(scores)

print("Mean:", mean_score)
print("Variance:", variance_score)
```

## Educational Impact

STMATH is intended to support undergraduate STEM education by reducing barriers to experimentation in scientific Python workflows. The toolkit may be useful in classroom demonstrations, introductory laboratory exercises, and self-guided learning environments where students benefit from simplified interfaces and rapid prototyping capabilities.

The project focuses on conceptual exploration, beginner-oriented experimentation, and scaffolded learning experiences that help students gradually transition toward broader scientific Python ecosystems.

STMATH is particularly targeted toward first- and second-year learners studying mathematics, statistics, scientific computing, and introductory artificial intelligence concepts.


## AI Usage Disclosure

AI-assisted tools were used for limited drafting and language refinement during manuscript preparation. All technical content, implementation decisions, and verification were performed and reviewed by the author.


## Limitations

STMATH is not intended to replace mature scientific computing ecosystems such as NumPy, SciPy, or scikit-learn for advanced research or production-scale workflows. The project is specifically designed for introductory educational settings, conceptual experimentation, and beginner-oriented scientific computing instruction.


## Acknowledgements

The author would like to acknowledge Medi-Caps University, Indore, for academic support. This work did not receive any specific grant from public, commercial, or not-for-profit funding agencies.

## References









 











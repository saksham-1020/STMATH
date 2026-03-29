# 🤝 Contributing to STMATH

Thank you for your interest in contributing to **STMATH** 🚀

STMATH is a **Zero-Wrapper Mathematical and AI Framework** built from first principles, combining:

* Mathematics
* Machine Learning
* Deep Learning
* NLP
* Computer Vision
* GenAI (Transformers)

---

## 🧠 Philosophy of Contribution

STMATH follows a **First-Principles + Zero-Wrapper Philosophy**:

✔ No black-box dependencies
✔ No reliance on `math`, `numpy`, or external numerical libraries for core logic
✔ All algorithms must be **educational, transparent, and reproducible**

👉 Every contribution must respect this philosophy.

---

## 📌 How to Contribute

1. **Fork the repository**

2. **Clone your fork**

   ```bash
   git clone https://github.com/your-username/stmath.git
   cd stmath
   ```

3. **Create a new branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**

   * Add feature / fix bug / improve docs

5. **Run tests**

   ```bash
   pytest
   ```

6. **Commit changes**

   ```bash
   git commit -m "feat: add <feature-name>"
   ```

7. **Push to GitHub**

   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open Pull Request**

   * Explain:

     * What you changed
     * Why it was needed
     * Expected impact

---

## 🧩 Contribution Types

We welcome:

* 🐛 Bug fixes
* ✨ New algorithms (ML, DL, NLP, Vision, GenAI)
* 📚 Documentation improvements
* ⚡ Performance optimizations
* 🧪 Test cases and validation

---

## ⚙️ Coding Guidelines

### 🔹 Core Rules (VERY IMPORTANT)

```text
❌ No math / numpy / torch usage in core logic
✔ Use custom math_kernels (sqrt, exp, log)
✔ Follow zero-wrapper principle
```

---

### 🔹 General Guidelines

* Write clean and readable Python code
* Follow **PEP8** style guidelines
* Add **docstrings** for all public functions
* Keep code **modular and educational**
* Avoid breaking existing APIs
* Add tests for new features

---

## 🧪 Testing

STMATH uses **pytest**.

Run tests:

```bash
pip install pytest
pytest
```

Before submitting PR:

```text
✔ All tests must pass
✔ No new warnings/errors
```

---

## 🔬 Validation Expectations

Every contribution should ensure:

* Numerical correctness
* Stability (use EPS where required)
* Convergence (for ML/DL models)
* Reproducibility

---

## 📊 Project Structure Awareness

Before contributing, understand:

```text
core/      → autograd, tensors, math kernels  
ml/        → machine learning models  
nn/        → neural networks  
nlp/       → text processing  
vision/    → image processing  
genai/     → transformer architecture  
utils/     → benchmarking, constants  
```

---

## 🏷 Issue Labels

| Label            | Meaning             |
| ---------------- | ------------------- |
| good first issue | Beginner-friendly   |
| help wanted      | Community needed    |
| bug              | Confirmed issue     |
| enhancement      | Feature improvement |

---

## 💬 Communication

* Use GitHub Issues for discussion
* Propose large changes before implementation
* Be respectful and constructive

---

## 🛡 Code of Conduct

All contributors must:

✔ Be respectful
✔ Be inclusive
✔ Maintain professionalism

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the **MIT License**.

---

## 💀 Final Note

STMATH is not just a library.

It is a:

✔ Mathematical Engine
✔ AI Framework
✔ Research System

👉 Every contribution should aim to **improve clarity, correctness, and depth**.

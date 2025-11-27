#  STMATH: Unified Math & AI Toolkit for Python

[![PyPI version](https://img.shields.io/pypi/v/stmath.svg)](https://pypi.org/project/stmath/)
[![Python](https://img.shields.io/pypi/pyversions/stmath.svg)](https://pypi.org/project/stmath/)
[![License](https://img.shields.io/github/license/saksham-1020/STMATH.svg)](https://github.com/saksham-1020/STMATH/blob/main/LICENSE)


##  What is STMATH?

**STMATH** is a modular, educational, and developer-friendly Python library for mathematics, AI, ML, quantum computing, cryptography, vision, graph algorithms, and GenAI helpers. It’s designed for researchers, students, and educators who want clean, reusable functions with perfect documentation and publishing clarity.

---

## 📚 Table of Contents

1. [Main Features](#main-features)  
2. [Installation](#installation)  
3. [Domains Covered](#domains-covered)  
4. [ <span style="color:#007acc"><strong>License</strong></span>](#license)  
5. [Documentation](#documentation)  
6. [Contributing](#contributing)  
7. [Benchmarks](#benchmarks)  
8. [About STMATH](#about-stmath)

---

##  Main Features

STMATH offers:

1. Statistics & Probability: mean, median, mode, variance, distributions, Bayes, z-score  
2. ML Metrics: accuracy, precision, recall, F1, confusion matrix, regression scores  
3. Deep Learning: relu, sigmoid, softmax, entropy, cross-entropy, KL divergence  
4. GenAI Math: logits→prob, softmax with temperature, attention scores  
5. Cryptography: SHA256, gas fee calculator, modular inverse, totient, primes  
6. Quantum: Hadamard, Pauli gates, quantum state transforms  
7. Graph Theory: BFS, Dijkstra, shortest paths, adjacency utilities  
8. Time Series: SMA, EMA, moving averages  
9. Number Theory: gcd, lcm, primes, totient, Fibonacci, Catalan, divisor tools  
10. Vision: conv2d/maxpool shapes, IoU, NMS  
11. Optimization: SGD, Adam, RMSProp, cosine annealing, momentum updates  
12. Finance: interest, EMI, ROI, CAGR  
13. Aptitude: profit/loss %, average speed, work/time problems  
14. Benchmarking: timeit, memory profile, performance comparison  
15. Math Extensions: factorial, abs, round, floor, ceil, sign, clamp, pow10

---

###  Domains Covered

- Core Math & Scientific Functions  
- Statistics & Probability  
- ML & DL Metrics  
- GEN-AI Math  
- Graph Algorithms  
- Vision Utilities  
- Optimizers  
- Finance & Aptitude  
- Cryptography  
- Quantum Computing  
- Time Series Analysis  
- Number Theory  
- Benchmarking

##  Installation

###  First-Time Install (Jupyter/Colab)
```python
!pip install stmath
```
##  Upgrade to Latest Version
```python
!pip install --upgrade stmath
```
##  Function Handbook Examples

###  Core Math
```python
import stmath as am

print(am.add(10, 5))     # → 15
print(am.sub(10, 5))     # → 5
print(am.mul(10, 5))     # → 50
print(am.div(10, 5))     # → 2.0
print(am.square(4))      # → 16
print(am.cube(3))        # → 27
print(am.sqrt(16))       # → 4.0
print(am.power(2, 3))    # → 8
```
##  Core Math Functions

1. **add(a, b)**  
   - Syntax: `am.add(a, b)`  
   - Example: `am.add(10, 5)` → 15  
   - Formula: a + b  

2. **sub(a, b)**  
   - Syntax: `am.sub(a, b)`  
   - Example: `am.sub(10, 5)` → 5  
   - Formula: a − b  

3. **mul(a, b)**  
   - Syntax: `am.mul(a, b)`  
   - Example: `am.mul(10, 5)` → 50  
   - Formula: a × b  

4. **div(a, b)**  
   - Syntax: `am.div(a, b)`  
   - Example: `am.div(10, 5)` → 2.0  
   - Formula: a ÷ b  

5. **square(x)**  
   - Syntax: `am.square(x)`  
   - Example: `am.square(4)` → 16  
   - Formula: x²  

6. **cube(x)**  
   - Syntax: `am.cube(x)`  
   - Example: `am.cube(3)` → 27  
   - Formula: x³  

7. **sqrt(x)**  
   - Syntax: `am.sqrt(x)`  
   - Example: `am.sqrt(16)` → 4.0  
   - Formula: √x  

8. **power(x, y)**  
   - Syntax: `am.power(x, y)`  
   - Example: `am.power(2, 3)` → 8  
   - Formula: xʸ  

9. **percent(part, whole)**  
   - Syntax: `am.percent(part, whole)`  
   - Example: `am.percent(50, 200)` → 25.0  
   - Formula: (part ÷ whole) × 100  

10. **percent_change(old, new)**  
    - Syntax: `am.percent_change(old, new)`  
    - Example: `am.percent_change(100, 120)` → 20.0  
    - Formula: (new − old) ÷ old × 100  
```
###  Scientific Functions
```python
import stmath as am

print(am.exp(1))          # → 2.718 (Euler’s number e^1)
print(am.log(10))         # → 2.302 (Natural log)
print(am.log10(100))      # → 2.0 (Base‑10 log)
print(am.sin(3.14))       # → 0.00159
print(am.cos(3.14))       # → -1.0
print(am.tan(0.785))      # → 1.0
print(am.sinh(1))         # → 1.175
print(am.cosh(1))         # → 1.543
print(am.tanh(1))         # → 0.761
print(am.deg2rad(180))    # → 3.14159
print(am.rad2deg(3.14159))# → 180.0
```
##  Scientific Functions

1. **sin(x)**  
   - Syntax: `am.sin(x)`  
   - Example: `am.sin(am.pi/2)` → 1.0  
   - Formula: sin(x)  

2. **cos(x)**  
   - Syntax: `am.cos(x)`  
   - Example: `am.cos(0)` → 1.0  
   - Formula: cos(x)  

3. **tan(x)**  
   - Syntax: `am.tan(x)`  
   - Example: `am.tan(am.pi/4)` → 1.0  
   - Formula: tan(x)  

4. **log10(x)**  
   - Syntax: `am.log10(x)`  
   - Example: `am.log10(100)` → 2.0  
   - Formula: log₁₀(x)  

5. **ln(x)**  
   - Syntax: `am.ln(x)`  
   - Example: `am.ln(am.e)` → 1.0  
   - Formula: ln(x)  

6. **exp(x)**  
   - Syntax: `am.exp(x)`  
   - Example: `am.exp(1)` → 2.718…  
   - Formula: eˣ  

7. **factorial(n)**  
   - Syntax: `am.factorial(n)`  
   - Example: `am.factorial(5)` → 120  
   - Formula: n! = 1 × 2 × 3 … × n  

8. **gcd(a, b)**  
   - Syntax: `am.gcd(a, b)`  
   - Example: `am.gcd(12, 18)` → 6  
   - Formula: gcd(a, b)  

9. **lcm(a, b)**  
   - Syntax: `am.lcm(a, b)`  
   - Example: `am.lcm(12, 18)` → 36  
   - Formula: lcm(a, b)  

10. **deg2rad(deg)**  
    - Syntax: `am.deg2rad(deg)`  
    - Example: `am.deg2rad(180)` → 3.14159  
    - Formula: (π ÷ 180) × deg  

11. **rad2deg(rad)**  
    - Syntax: `am.rad2deg(rad)`  
    - Example: `am.rad2deg(am.pi)` → 180  
    - Formula: (180 ÷ π) × rad  
```
###  Probability & Statistics
```python
import stmath as am

print(am.mean([1,2,3,4,5]))        # → 3.0
print(am.variance([1,2,3,4,5]))    # → 2.5
print(am.std_dev([1,2,3,4,5]))     # → 1.58
print(am.binomial_pmf(n=5, k=2, p=0.5))  # → 0.3125
print(am.normal_pdf(x=0, mean=0, sd=1))  # → 0.3989
```
##  Statistics Functions

1. **mean(data)**  
   - Syntax: `am.mean(data)`  
   - Example: `am.mean([10, 20, 30])` → 20  
   - Formula: (Σxᵢ) ÷ n  

2. **median(data)**  
   - Syntax: `am.median(data)`  
   - Example: `am.median([10, 20, 30])` → 20  
   - Formula: middle value of sorted data  

3. **mode(data)**  
   - Syntax: `am.mode(data)`  
   - Example: `am.mode([10, 20, 20, 30])` → 20  
   - Formula: most frequent value  

4. **variance(data)**  
   - Syntax: `am.variance(data)`  
   - Example: `am.variance([10, 20, 30])` → 66.67  
   - Formula: Σ(xᵢ − μ)² ÷ n  

5. **std(data)**  
   - Syntax: `am.std(data)`  
   - Example: `am.std([10, 20, 30])` → 8.16  
   - Formula: √variance  

6. **data_range(data)**  
   - Syntax: `am.data_range(data)`  
   - Example: `am.data_range([10, 20, 30])` → 20  
   - Formula: max − min  

7. **iqr(data)**  
   - Syntax: `am.iqr(data)`  
   - Example: `am.iqr([1, 2, 3, 4, 5, 6, 7, 8])` → 4  
   - Formula: Q3 − Q1  

8. **z_score(x, mean, std)**  
   - Syntax: `am.z_score(x, mean, std)`  
   - Example: `am.z_score(70, 60, 5)` → 2.0  
   - Formula: (x − μ) ÷ σ  
``
##  Probability & Distributions Functions

1. **nCr(n, r)**  
   - Syntax: `am.nCr(n, r)`  
   - Example: `am.nCr(5, 2)` → 10  
   - Formula: n! ÷ (r! × (n − r)!)  

2. **nPr(n, r)**  
   - Syntax: `am.nPr(n, r)`  
   - Example: `am.nPr(5, 2)` → 20  
   - Formula: n! ÷ (n − r)!  

3. **bayes(PA, PB, PBA)**  
   - Syntax: `am.bayes(PA, PB, PBA)`  
   - Example: `am.bayes(0.5, 0.4, 0.7)` → 0.875  
   - Formula: P(A|B) = (P(B|A) × P(A)) ÷ P(B)  

4. **expected_value(values, probs)**  
   - Syntax: `am.expected_value(values, probs)`  
   - Example: `am.expected_value([1,2,3],[0.2,0.3,0.5])` → 2.3  
   - Formula: Σ(xᵢ × pᵢ)  

5. **normal_pdf(x, μ, σ)**  
   - Syntax: `am.normal_pdf(x, μ, σ)`  
   - Example: `am.normal_pdf(0, 0, 1)` → 0.3989  
   - Formula: (1 ÷ (σ√2π)) × e^(−(x − μ)² ÷ (2σ²))  

6. **normal_cdf(x, μ, σ)**  
   - Syntax: `am.normal_cdf(x, μ, σ)`  
   - Example: `am.normal_cdf(0, 0, 1)` → 0.5  
   - Formula: cumulative distribution of normal  

7. **bernoulli_pmf(k, p)**  
   - Syntax: `am.bernoulli_pmf(k, p)`  
   - Example: `am.bernoulli_pmf(1, 0.6)` → 0.6  
   - Formula: pᵏ × (1 − p)^(1 − k)  

8. **binomial_pmf(k, n, p)**  
   - Syntax: `am.binomial_pmf(k, n, p)`  
   - Example: `am.binomial_pmf(2, 5, 0.5)` → 0.3125  
   - Formula: (nCr(n, k)) × pᵏ × (1 − p)^(n − k)  

9. **poisson_pmf(k, λ)**  
   - Syntax: `am.poisson_pmf(k, λ)`  
   - Example: `am.poisson_pmf(3, 2)` → 0.1804  
   - Formula: (e^(−λ) × λᵏ) ÷ k!  

10. **exponential_pdf(x, λ)**  
    - Syntax: `am.exponential_pdf(x, λ)`  
    - Example: `am.exponential_pdf(2, 1)` → 0.1353  
    - Formula: λ × e^(−λx)  

11. **uniform_pdf(a, b)**  
    - Syntax: `am.uniform_pdf(a, b)`  
    - Example: `am.uniform_pdf(0, 5)` → 0.2  
    - Formula: 1 ÷ (b − a)  

12. **t_pdf(x, ν)**  
    - Syntax: `am.t_pdf(x, ν)`  
    - Example: `am.t_pdf(0, 10)` → 0.389  
    - Formula: Student’s t-distribution formula  

13. **chi_square_pdf(x, k)**  
    - Syntax: `am.chi_square_pdf(x, k)`  
    - Example: `am.chi_square_pdf(2, 4)` → 0.151  
    - Formula: Chi-square distribution formula  
``
###  Machine Learning Metrics
```python
import stmath as am

y_true = [1,0,1,1]
y_pred = [1,0,0,1]

print(am.accuracy(y_true, y_pred))   # → 0.75
print(am.precision(y_true, y_pred))  # → 1.0
print(am.recall(y_true, y_pred))     # → 0.66
print(am.f1_score(y_true, y_pred))   # → 0.8
```
##  Machine Learning Metrics Functions

1. **accuracy(y_true, y_pred)**  
   - Syntax: `am.accuracy(y_true, y_pred)`  
   - Example: `am.accuracy([1,0,1,1],[1,0,0,1])` → 0.75  
   - Formula: (TP + TN) ÷ (TP + TN + FP + FN)  

2. **precision(y_true, y_pred)**  
   - Syntax: `am.precision(y_true, y_pred)`  
   - Example: `am.precision([1,0,1,1],[1,0,0,1])` → 1.0  
   - Formula: TP ÷ (TP + FP)  

3. **recall(y_true, y_pred)**  
   - Syntax: `am.recall(y_true, y_pred)`  
   - Example: `am.recall([1,0,1,1],[1,0,0,1])` → 0.666…  
   - Formula: TP ÷ (TP + FN)  

4. **f1_score(y_true, y_pred)**  
   - Syntax: `am.f1_score(y_true, y_pred)`  
   - Example: `am.f1_score([1,0,1,1],[1,0,0,1])` → 0.8  
   - Formula: 2 × (precision × recall) ÷ (precision + recall)  

5. **confusion_matrix(y_true, y_pred)**  
   - Syntax: `am.confusion_matrix(y_true, y_pred)`  
   - Example: `am.confusion_matrix([1,0,1,1],[1,0,0,1])` → [[1,0],[1,2]]  
   - Formula: matrix of TP, TN, FP, FN counts  

6. **mse(y_true, y_pred)**  
   - Syntax: `am.mse(y_true, y_pred)`  
   - Example: `am.mse([1,2,3],[1,2,4])` → 0.333…  
   - Formula: Σ(yᵢ − ŷᵢ)² ÷ n  

7. **rmse(y_true, y_pred)**  
   - Syntax: `am.rmse(y_true, y_pred)`  
   - Example: `am.rmse([1,2,3],[1,2,4])` → 0.577…  
   - Formula: √MSE  

8. **mae(y_true, y_pred)**  
   - Syntax: `am.mae(y_true, y_pred)`  
   - Example: `am.mae([1,2,3],[1,2,4])` → 0.333…  
   - Formula: Σ|yᵢ − ŷᵢ| ÷ n  

9. **r2_score(y_true, y_pred)**  
   - Syntax: `am.r2_score(y_true, y_pred)`  
   - Example: `am.r2_score([1,2,3],[1,2,4])` → 0.9  
   - Formula: 1 − (Σ(yᵢ − ŷᵢ)² ÷ Σ(yᵢ − ȳ)²)  
   ###  Deep Learning Functions
```python
import stmath as am

y_true = [1,0,1,1]
y_pred = [1,0,0,1]

print(am.accuracy(y_true, y_pred))   # → 0.75
print(am.precision(y_true, y_pred))  # → 1.0
print(am.recall(y_true, y_pred))     # → 0.66
print(am.f1_score(y_true, y_pred))   # → 0.8
```
##  Deep Learning Functions

1. **relu(x)**  
   - Syntax: `am.relu(x)`  
   - Example: `am.relu(-5)` → 0  
   - Formula: max(0, x)  

2. **sigmoid(x)**  
   - Syntax: `am.sigmoid(x)`  
   - Example: `am.sigmoid(0)` → 0.5  
   - Formula: 1 ÷ (1 + e^(−x))  

3. **tanh(x)**  
   - Syntax: `am.tanh(x)`  
   - Example: `am.tanh(0)` → 0.0  
   - Formula: (e^x − e^(−x)) ÷ (e^x + e^(−x))  

4. **softmax(values)**  
   - Syntax: `am.softmax(values)`  
   - Example: `am.softmax([1,2,3])` → [0.09, 0.24, 0.66]  
   - Formula: e^(xᵢ) ÷ Σ(e^(xⱼ))  

5. **entropy(probs)**  
   - Syntax: `am.entropy(probs)`  
   - Example: `am.entropy([0.5,0.5])` → 0.693  
   - Formula: −Σ(pᵢ × log(pᵢ))  

6. **kl_divergence(p, q)**  
   - Syntax: `am.kl_divergence(p, q)`  
   - Example: `am.kl_divergence([0.5,0.5],[0.9,0.1])` → 0.51  
   - Formula: Σ(pᵢ × log(pᵢ ÷ qᵢ))  

7. **binary_cross_entropy(y_true, y_pred)**  
   - Syntax: `am.binary_cross_entropy(y_true, y_pred)`  
   - Example: `am.binary_cross_entropy([1,0],[0.9,0.1])` → 0.105  
   - Formula: −[y × log(ŷ) + (1 − y) × log(1 − ŷ)]  


###  GenAI Math
```python
import stmath as am

logits = [2.0, 1.0, 0.1]
print(am.softmax(logits))                  # → [0.659, 0.242, 0.099]
print(am.temperature_softmax(logits, T=2)) # smoother distribution
print(am.attention([0.2,0.3,0.5]))         # → normalized weights
```
##  GEN-AI Math Functions

1. **logits_to_prob(logits)**  
   - Syntax: `am.logits_to_prob(logits)`  
   - Example: `am.logits_to_prob([2.0, 1.0, 0.1])` → [0.659, 0.242, 0.099]  
   - Formula: Convert raw logits → probabilities using softmax normalization  

2. **softmax_temperature(logits, T)**  
   - Syntax: `am.softmax_temperature(logits, T)`  
   - Example: `am.softmax_temperature([2.0, 1.0, 0.1], T=2.0)` → [0.45, 0.30, 0.25]  
   - Formula: e^(xᵢ/T) ÷ Σ(e^(xⱼ/T))  
   - Note: Higher T → smoother distribution, Lower T → sharper distribution  

3. **attention_scores(weights)**  
   - Syntax: `am.attention_scores(weights)`  
   - Example: `am.attention_scores([0.2, 0.3, 0.5])` → [0.2, 0.3, 0.5] (normalized)  
   - Formula: Normalize weights so Σ = 1 (softmax‑style normalization)  

###  Cryptography
```python
import stmath as am

print(am.sha256("hello"))  
# → "2cf24dba5fb0a30e26e83b2ac5b9e29e..."

print(am.gas_fee(gas_used=21000, gwei=50, eth_price=2000))  
# → 2.1 USD (approx)
```
##  Cryptography Functions

1. **sha256(text)**  
   - Syntax: `am.sha256(text)`  
   - Example: `am.sha256("hello")` → `"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"`  
   - Formula: SHA‑256 cryptographic hash of input text  

2. **gas_fee(gas_used, gwei, eth_price)**  
   - Syntax: `am.gas_fee(gas_used, gwei, eth_price)`  
   - Example: `am.gas_fee(21000, 50, 2000)` → 2.1  
   - Formula: (gas_used × gwei × 1e‑9) × eth_price (in USD)  

###  Quantum Functions
```python
import stmath as am

print(am.hadamard([1,0]))   # → [0.707, 0.707]
print(am.pauli_x([1,0]))    # → [0,1]
print(am.pauli_z([1,0]))    # → [1,0]
```
##  Quantum Functions

1. **hadamard(state)**  
   - Syntax: `am.hadamard(state)`  
   - Example: `am.hadamard([1,0])` → [0.707, 0.707]  
   - Formula: H|0⟩ = (|0⟩ + |1⟩) ÷ √2  

2. **pauli_x(state)**  
   - Syntax: `am.pauli_x(state)`  
   - Example: `am.pauli_x([1,0])` → [0,1]  
   - Formula: X|0⟩ = |1⟩, X|1⟩ = |0⟩  

###  Graph Algorithms
```python
import stmath as am

adj = {
    0: [1,2],
    1: [2],
    2: [0,3],
    3: [3]
}

print(am.bfs_distance(adj, 0))            # → {0:0, 1:1, 2:1, 3:2}
print(am.dijkstra_shortest_path(adj, 0))  # shortest paths
```
##  Graph Theory Functions

1. **bfs_distance(adj, start)**  
   - Syntax: `am.bfs_distance(adj, start)`  
   - Example:  
     ```python
     am.bfs_distance({"A":["B","C"],"B":["D"],"C":[],"D":[]}, "A")
     ```  
     → `{"A":0,"B":1,"C":1,"D":2}`  
   - Formula: breadth‑first search distance from start node  

2. **dijkstra_shortest_path(adj, start)**  
   - Syntax: `am.dijkstra_shortest_path(adj, start)`  
   - Example:  
     ```python
     am.dijkstra_shortest_path({"A":["B","C"],"B":["C"],"C":[]}, "A")
     ```  
     → `({"A":0,"B":1,"C":1}, {"A":None,"B":"A","C":"A"})`  
   - Formula: shortest path distances from start node using Dijkstra’s algorithm (default weight = 1)  
    
###  Time Series
```python
import stmath as am

data = [1,2,3,4,5]

print(am.sma(data, 3))   # → [None, None, 2.0, 3.0, 4.0]
print(am.ema(data, 0.5)) # → exponential moving average
```
##  Time Series Functions

1. **sma(data, window)**  
   - Syntax: `am.sma(data, window)`  
   - Example: `am.sma([1,2,3,4,5], 3)` → [2.0, 3.0, 4.0]  
   - Formula: average of last *window* values  

2. **ema(data, alpha)**  
   - Syntax: `am.ema(data, alpha)`  
   - Example: `am.ema([1,2,3,4], 0.5)` → [1, 1.5, 2.25, 3.125]  
   - Formula: EMAₜ = αxₜ + (1 − α)EMAₜ₋₁  

###  Number Theory
```python
import stmath as am

print(am.gcd(48, 18))          # → 6
print(am.lcm(12, 15))          # → 60
print(am.is_prime(29))         # → True
print(am.mod_inverse(3, 11))   # → 4
print(am.fibonacci(10))        # → 55
```
##  Number Theory Functions

1. **gcd(a, b)**  
   - Syntax: `am.gcd(a, b)`  
   - Example: `am.gcd(12, 18)` → 6  
   - Formula: greatest common divisor  

2. **lcm(a, b)**  
   - Syntax: `am.lcm(a, b)`  
   - Example: `am.lcm(12, 18)` → 36  
   - Formula: least common multiple  

3. **is_prime(n)**  
   - Syntax: `am.is_prime(n)`  
   - Example: `am.is_prime(17)` → True  
   - Formula: checks primality  

4. **prime_factors(n)**  
   - Syntax: `am.prime_factors(n)`  
   - Example: `am.prime_factors(28)` → [2, 2, 7]  
   - Formula: factorization into primes  

5. **totient(n)**  
   - Syntax: `am.totient(n)`  
   - Example: `am.totient(9)` → 6  
   - Formula: Euler’s totient function  

6. **mod_inverse(a, m)**  
   - Syntax: `am.mod_inverse(a, m)`  
   - Example: `am.mod_inverse(3, 11)` → 4  
   - Formula: a⁻¹ mod m  

7. **modular_pow(base, exp, mod)**  
   - Syntax: `am.modular_pow(base, exp, mod)`  
   - Example: `am.modular_pow(2, 10, 1000)` → 24  
   - Formula: (base^exp) mod m  

8. **fibonacci(n)**  
   - Syntax: `am.fibonacci(n)`  
   - Example: `am.fibonacci(10)` → 55  
   - Formula: nth Fibonacci number  

9. **pell_number(n)**  
   - Syntax: `am.pell_number(n)`  
   - Example: `am.pell_number(5)` → 29  
   - Formula: recurrence Pₙ = 2Pₙ₋₁ + Pₙ₋₂  

10. **catalan_number(n)**  
    - Syntax: `am.catalan_number(n)`  
    - Example: `am.catalan_number(4)` → 14  
    - Formula: (1 ÷ (n+1)) × (2n choose n)  

11. **divisor_count(n)**  
    - Syntax: `am.divisor_count(n)`  
    - Example: `am.divisor_count(12)` → 6  
    - Formula: number of divisors  

12. **divisor_sum(n)**  
    - Syntax: `am.divisor_sum(n)`  
    - Example: `am.divisor_sum(12)` → 28  
    - Formula: sum of divisors  

###  Vision Utilities
```python
import stmath as am

print(am.conv2d_output(h=32, w=32, k=3, stride=1, pad=0))  
# → (30, 30)

print(am.maxpool_shape(h=32, w=32, k=2, stride=2))  
# → (16, 16)

print(am.iou([0,0,10,10], [5,5,15,15]))  
# → 0.1428

print(am.nms([[0,0,10,10,0.9],[1,1,9,9,0.8]], threshold=0.5))  
# → keep highest confidence box
```
##  Vision Functions

1. **conv2d_output_shape(input_shape, kernel, stride, padding)**  
   - Syntax: `am.conv2d_output_shape(input_shape, kernel, stride, padding)`  
   - Example: `am.conv2d_output_shape((28,28), (3,3), (1,1), (0,0))` → (26,26)  
   - Formula: ((W − K + 2P) ÷ S + 1, (H − K + 2P) ÷ S + 1)  

2. **maxpool_output_shape(input_shape, pool, stride)**  
   - Syntax: `am.maxpool_output_shape(input_shape, pool, stride)`  
   - Example: `am.maxpool_output_shape((28,28), (2,2), (2,2))` → (14,14)  
   - Formula: ((W − P) ÷ S + 1, (H − P) ÷ S + 1)  

3. **iou(box1, box2)**  
   - Syntax: `am.iou(box1, box2)`  
   - Example: `am.iou([0,0,2,2],[1,1,3,3])` → 0.1428  
   - Formula: intersection area ÷ union area  

4. **nms(boxes, threshold)**  
   - Syntax: `am.nms(boxes, threshold)`  
   - Example: `am.nms([[0,0,2,2,0.9],[1,1,3,3,0.8]], 0.5)` → [[0,0,2,2,0.9]]  
   - Formula: suppress overlapping boxes above threshold  

###  Optimizers
```python
import stmath as am

params = [0.5, -0.3]
grads = [0.1, -0.2]

print(am.sgd(params, grads, lr=0.01))       # → updated params
print(am.adam(params, grads, lr=0.01))      # → updated params
print(am.rmsprop(params, grads, lr=0.01))   # → updated params
print(am.cosine_anneal(lr=0.1, step=5, T=10)) # → annealed learning rate
```
##  Optimization Functions

1. **sgd_update(param, grad, lr)**  
   - Syntax: `am.sgd_update(param, grad, lr)`  
   - Example: `am.sgd_update(1.0, 0.1, 0.01)` → 0.999  
   - Formula: param − lr × grad  

2. **adam_update(param, grad, m, v, t, lr, beta1, beta2, eps)**  
   - Syntax: `am.adam_update(param, grad, m, v, t, lr, beta1, beta2, eps)`  
   - Example:  
     ```python
     am.adam_update(1.0, 0.1, 0, 0, 1, 0.01, 0.9, 0.999, 1e−8)
     ```  
     → updated param  
   - Formula: adaptive moment estimation update  

3. **rmsprop_update(param, grad, cache, lr, beta, eps)**  
   - Syntax: `am.rmsprop_update(param, grad, cache, lr, beta, eps)`  
   - Example:  
     ```python
     am.rmsprop_update(1.0, 0.1, 0, 0.01, 0.9, 1e−8)
     ```  
     → updated param  
   - Formula: RMSProp update rule  

4. **lr_step_decay(lr, step, decay)**  
   - Syntax: `am.lr_step_decay(lr, step, decay)`  
   - Example: `am.lr_step_decay(0.1, 10, 0.5)` → 0.05  
   - Formula: lr × decay^(step)  

5. **lr_cosine_anneal(lr, t, T)**  
   - Syntax: `am.lr_cosine_anneal(lr, t, T)`  
   - Example: `am.lr_cosine_anneal(0.1, 5, 10)` → 0.05  
   - Formula: lr × 0.5 × (1 + cos(πt ÷ T))  

6. **momentum_update(param, grad, velocity, lr, beta)**  
   - Syntax: `am.momentum_update(param, grad, velocity, lr, beta)`  
   - Example:  
     ```python
     am.momentum_update(1.0, 0.1, 0, 0.01, 0.9)
     ```  
     → updated param  
   - Formula: momentum gradient descent update  

###   Finance Math
```python
import stmath as am

print(am.simple_interest(p=1000, r=5, t=2))   # → 100.0
print(am.compound_interest(p=1000, r=5, t=2)) # → 102.5
print(am.emi(principal=500000, rate=7.5, years=20))  # → monthly EMI
print(am.future_value(p=1000, r=10, t=5))     # → 1610.51
```
##  Finance Functions

1. **simple_interest(principal, rate, time)**  
   - Syntax: `am.simple_interest(principal, rate, time)`  
   - Example: `am.simple_interest(1000, 5, 2)` → 100.0  
   - Formula: (P × R × T) ÷ 100  

2. **compound_interest(principal, rate, time)**  
   - Syntax: `am.compound_interest(principal, rate, time)`  
   - Example: `am.compound_interest(1000, 5, 2)` → 102.5  
   - Formula: P × (1 + R ÷ 100)^T − P  

3. **loan_emi(principal, rate, time)**  
   - Syntax: `am.loan_emi(principal, rate, time)`  
   - Example: `am.loan_emi(100000, 10, 12)` → 8791.59  
   - Formula: [P × R × (1+R)^T] ÷ [(1+R)^T − 1]  

###  Aptitude Math
```python
import stmath as am

print(am.percent(50, 200))             # → 25.0
print(am.percent_change(100, 120))     # → 20.0
print(am.ratio(2, 5))                  # → "2:5"
print(am.permutation(n=5, r=2))        # → 20
print(am.combination(n=5, r=2))        # → 10
```
##  Aptitude Functions

1. **profit_percent(cost, selling)**  
   - Syntax: `am.profit_percent(cost, selling)`  
   - Example: `am.profit_percent(100, 120)` → 20.0  
   - Formula: ((SP − CP) ÷ CP) × 100  

2. **loss_percent(cost, selling)**  
   - Syntax: `am.loss_percent(cost, selling)`  
   - Example: `am.loss_percent(100, 80)` → 20.0  
   - Formula: ((CP − SP) ÷ CP) × 100  

3. **avg_speed(distance1, speed1, distance2, speed2)**  
   - Syntax: `am.avg_speed(distance1, speed1, distance2, speed2)`  
   - Example: `am.avg_speed(60, 30, 60, 60)` → 40.0  
   - Formula: total distance ÷ total time  

###  Benchmarking Tools
```python
import stmath as am

# Time performance of a function
print(am.timeit(lambda: am.add(10, 5)))  

# Memory usage of a function
print(am.mem_profile(lambda: am.mul(1000, 2000)))  

# Compare two functions
print(am.compare_perf(lambda: am.add(10,5), lambda: am.mul(10,5)))
```
##  Benchmark Functions

1. **timeit(func, *args)**  
   - Syntax: `am.timeit(func, *args)`  
   - Example: `am.timeit(sum, [1,2,3])` → execution time  
   - Formula: measures runtime of function  

2. **mem_profile(func, *args)**  
   - Syntax: `am.mem_profile(func, *args)`  
   - Example: `am.mem_profile(sum, [1,2,3])` → memory usage  
   - Formula: measures memory usage of function  

###  Math Extensions
```python
import stmath as am

y_true = [1,0,1,1]
y_pred = [1,0,0,1]

print(am.factorial(5))         # → 120
print(am.abs_val(-42))         # → 42
print(am.round_val(3.14159, 2))# → 3.14
print(am.floor_val(3.9))       # → 3
print(am.ceil_val(3.1))        # → 4
print(am.sign(-10))            # → -1
```
##  Math Extension Functions

1. **factorial(n)**  
   - Syntax: `am.factorial(n)`  
   - Example: `am.factorial(5)` → 120  
   - Formula: n! = 1 × 2 × … × n  

2. **abs_val(x)**  
   - Syntax: `am.abs_val(x)`  
   - Example: `am.abs_val(-42)` → 42  
   - Formula: |x|  

3. **round_val(x, decimals)**  
   - Syntax: `am.round_val(x, decimals)`  
   - Example: `am.round_val(3.14159, 2)` → 3.14  
   - Formula: round(x, decimals)  

4. **floor_val(x)**  
   - Syntax: `am.floor_val(x)`  
   - Example: `am.floor_val(3.9)` → 3  
   - Formula: ⌊x⌋ (greatest integer ≤ x)  

5. **ceil_val(x)**  
   - Syntax: `am.ceil_val(x)`  
   - Example: `am.ceil_val(3.1)` → 4  
   - Formula: ⌈x⌉ (smallest integer ≥ x)  

6. **sign(x)**  
   - Syntax: `am.sign(x)`  
   - Example: `am.sign(-10)` → −1  
   - Formula: returns −1 if x < 0, 0 if x = 0, 1 if x > 0  

7. **max_val(values)**  
   - Syntax: `am.max_val(values)`  
   - Example: `am.max_val([1, 5, 3])` → 5  
   - Formula: maximum element in list  

8. **min_val(values)**  
   - Syntax: `am.min_val(values)`  
   - Example: `am.min_val([1, 5, 3])` → 1  
   - Formula: minimum element in list  

9. **clamp(x, low, high)**  
   - Syntax: `am.clamp(x, low, high)`  
   - Example: `am.clamp(15, 0, 10)` → 10  
   - Formula: restricts x within [low, high]  

10. **pow10(n)**  
    - Syntax: `am.pow10(n)`  
    - Example: `am.pow10(3)` → 1000  
    - Formula: 10ⁿ  



---

## 🧩 Unique Highlights

- Unified Math + AI + Quantum + Crypto + Vision  
- Handbook-style docs with syntax, examples, formulas  
- Bilingual support (English + Hindi)  
- Safe testing via `safe_run(...)`  
- Educational clarity + developer performance  

---

## 🪪 License

MIT — free for personal, academic, and commercial use.

---

## 📖 Documentation

STMATH follows a handbook-style documentation approach — every function includes syntax, example, and formula.  
Full documentation will be hosted soon on GitHub Pages or ReadTheDocs.

For now, explore examples in this README or use `help(am.function_name)`.

Coming soon:
- Domain-wise docs with emojis and bilingual support  
- Visual examples for GenAI, Quantum, and Vision  
- Test coverage and `safe_run` wrappers  

---

## 🤝 Contributing

STMATH welcomes contributions from developers, educators, and researchers.

To contribute:
1. Fork the repo  
2. Create a feature branch  
3. Add tests and examples  
4. Submit a pull request  

Please follow the [CONTRIBUTING.md](./CONTRIBUTING.md) guide for standards and structure.

You can also:
- Open issues for bugs or feature requests  
- Share educational use-cases or notebooks  
- Help improve bilingual documentation  

---

## ℹ️ About STMATH

STMATH is built by [Saksham Tomar](https://www.linkedin.com/in/saksham-tomar), a Python developer and open-source educator.  
It aims to unify math, AI, and GenAI tooling into a single, clean, and reusable Python library.

Goals:
- Make math and AI accessible to learners  
- Provide reusable functions for research and education  
- Maintain professional publishing standards (PyPI + GitHub)  
- Support bilingual documentation (English + Hindi)  

If you use STMATH in your project or classroom, feel free to share and star the repo!

---

## 🔗 Project Links

- 📄 [PyPI Package](https://pypi.org/project/stmath)  
- 💼 [LinkedIn Profile](https://www.linkedin.com/in/saksham-tomar)  
- 💻 [GitHub Repo](https://github.com/saksham-1020/STMATH)
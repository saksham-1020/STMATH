# import math
# from typing import Iterable, List, Union
# from .special import fast_ln, exp_custom, sqrt_custom

# class Combinatorics:
#     @staticmethod
#     def log_factorial(n: int) -> float:
#         if n < 0: raise ValueError("Negative factorial")
#         if n < 2: return 0.0
#         # High-precision Stirling's Approximation
#         return n * fast_ln(n) - n + 0.5 * fast_ln(2 * math.pi * n)

#     @staticmethod
#     def nCr(n: int, r: int) -> int:
#         if r < 0 or r > n: return 0
#         if r == 0 or r == n: return 1
#         if r > n // 2: r = n - r
#         log_val = Combinatorics.log_factorial(n) - (Combinatorics.log_factorial(r) + Combinatorics.log_factorial(n - r))
#         return int(exp_custom(log_val) + 0.5)

#     @staticmethod
#     def nPr(n: int, r: int) -> int:
#         if r < 0 or r > n: return 0
#         log_val = Combinatorics.log_factorial(n) - Combinatorics.log_factorial(n - r)
#         return int(exp_custom(log_val) + 0.5)

# class Probability:
#     @staticmethod
#     def bayes(p_a, p_b, p_b_given_a):
#         if p_b == 0: raise ZeroDivisionError("P(B) zero")
#         return (p_b_given_a * p_a) / p_b

#     @staticmethod
#     def binomial_pdf(n, k, p):
#         log_val = Combinatorics.log_factorial(n) - (Combinatorics.log_factorial(k) + \
#                   Combinatorics.log_factorial(n - k)) + (k * fast_ln(p)) + ((n - k) * fast_ln(1 - p))
#         return exp_custom(log_val)

#     @staticmethod
#     def poisson_pdf(lmbda, k):
#         log_val = k * fast_ln(lmbda) - lmbda - Combinatorics.log_factorial(k)
#         return exp_custom(log_val)

#     @staticmethod
#     def normal_pdf(x, mu, sigma):
#         ln2pi = fast_ln(2 * math.pi)
#         log_val = -0.5 * (ln2pi + 2 * fast_ln(sigma) + ((x - mu) / sigma)**2)
#         return exp_custom(log_val)

# class InformationTheory:
#     @staticmethod
#     def entropy(p: Iterable[float], base: float = 2.0) -> float:
#         h = 0.0
#         ln_base = fast_ln(base)
#         for pi in p:
#             if pi > 0: h -= pi * (fast_ln(pi) / ln_base)
#         return h

#     @staticmethod
#     def kl_divergence(p: Iterable[float], q: Iterable[float], base: float = 2.0) -> float:
#         ln_base = fast_ln(base)
#         d_kl = 0.0
#         for pi, qi in zip(p, q):
#             if pi > 0:
#                 if qi <= 0: return float('inf')
#                 d_kl += pi * (fast_ln(pi / qi) / ln_base)
#         return d_kl

#     @staticmethod
#     def mutual_information(p_xy: List[List[float]], p_x: List[float], p_y: List[float]) -> float:
#         mi = 0.0
#         ln2 = fast_ln(2.0)
#         for i in range(len(p_x)):
#             for j in range(len(p_y)):
#                 if p_xy[i][j] > 0:
#                     mi += p_xy[i][j] * (fast_ln(p_xy[i][j] / (p_x[i] * p_y[j])) / ln2)
#         return mi
    
#     @staticmethod
#     def cross_entropy(p, q):
#         # H(P, Q) = H(P) + D_KL(P || Q)
#         return InformationTheory.entropy(p) + InformationTheory.kl_divergence(p, q)
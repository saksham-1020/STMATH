# from .special import exp_custom, fast_ln, sqrt_custom, pi, e
# from .number_theory import NumberTheory # Reusing GCD/LCM

# class Scientific:
#     # --- Trigonometry (Taylor Series Implementation) ---
#     @staticmethod
#     def sin(x: float, terms: int = 15) -> float:
#         """Sine implementation using Taylor Series: x - x^3/3! + x^5/5! ..."""
#         x = x % (2 * pi) # Bring within range
#         res = 0
#         for n in range(terms):
#             sign = (-1)**n
#             # x^(2n+1) / (2n+1)!
#             term = (x**(2*n + 1)) / Scientific.factorial(2*n + 1)
#             res += sign * term
#         return res

#     @staticmethod
#     def cos(x: float, terms: int = 15) -> float:
#         """Cosine implementation: 1 - x^2/2! + x^4/4! ..."""
#         x = x % (2 * pi)
#         res = 0
#         for n in range(terms):
#             sign = (-1)**n
#             term = (x**(2*n)) / Scientific.factorial(2*n)
#             res += sign * term
#         return res

#     @staticmethod
#     def tan(x: float) -> float:
#         c = Scientific.cos(x)
#         if abs(c) < 1e-15: return float('inf')
#         return Scientific.sin(x) / c

#     # --- Logarithms & Exponentials ---
#     @staticmethod
#     def ln(x: float) -> float:
#         return fast_ln(x) # Reusing our high-performance kernel

#     @staticmethod
#     def log10(x: float) -> float:
#         # log10(x) = ln(x) / ln(10)
#         return fast_ln(x) / fast_ln(10.0)

#     @staticmethod
#     def exp(x: float) -> float:
#         return exp_custom(x) # Reusing Newton-Raphson kernel

#     # --- Helpers ---
#     @staticmethod
#     def factorial(n: int) -> int:
#         if n < 0: raise ValueError("Not defined for negative")
#         if n == 0: return 1
#         res = 1
#         for i in range(2, n + 1):
#             res *= i
#         return res

#     @staticmethod
#     def deg2rad(deg: float) -> float:
#         return deg * (pi / 180.0)

#     @staticmethod
#     def rad2deg(rad: float) -> float:
#         return rad * (180.0 / pi)

#     # Re-exporting from NumberTheory to avoid redundancy
#     gcd = NumberTheory.gcd
#     lcm = NumberTheory.lcm
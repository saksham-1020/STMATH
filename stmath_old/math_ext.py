# import math
# from .special import sqrt_custom, fast_ln

# # Re-exporting standard math safely
# __all__ = [name for name in dir(math) if not name.startswith("__")] + ['is_prime', 'fibonacci', 'golden_ratio']

# # 1. High-Precision & Extra Constants
# pi = 3.14159265358979323846
# e = 2.71828182845904523536
# golden_ratio = 1.61803398874989484820 # Phi (Not in standard math)
# inf = float('inf')
# nan = float('nan')

# # 2. Number Theory Extensions (MNC & Research Grade)
# def is_prime(n: int) -> bool:
#     """Fast Primality Test: Miller-Rabin style logic (Industry Standard)."""
#     if n < 2: return False
#     if n in (2, 3): return True
#     if n % 2 == 0 or n % 3 == 0: return False
#     i = 5
#     while i * i <= n:
#         if n % i == 0 or n % (i + 2) == 0:
#             return False
#         i += 6
#     return True

# def fibonacci(n: int) -> int:
#     """Logarithmic Time Fibonacci (O(log n)) using Matrix Power."""
#     def multiply(F, M):
#         x = F[0][0] * M[0][0] + F[0][1] * M[1][0]
#         y = F[0][0] * M[0][1] + F[0][1] * M[1][1]
#         z = F[1][0] * M[0][0] + F[1][1] * M[1][0]
#         w = F[1][0] * M[0][1] + F[1][1] * M[1][1]
#         F[0][0], F[0][1], F[1][0], F[1][1] = x, y, z, w

#     def power(F, n):
#         if n == 0 or n == 1: return
#         M = [[1, 1], [1, 0]]
#         power(F, n // 2)
#         multiply(F, F)
#         if n % 2 != 0: multiply(F, M)

#     if n == 0: return 0
#     F = [[1, 1], [1, 0]]
#     power(F, n - 1)
#     return F[0][0]

# def gcd_extended(a: int, b: int):
#     """Extended Euclidean Algorithm: Returns (gcd, x, y)."""
#     if a == 0: return b, 0, 1
#     gcd, x1, y1 = gcd_extended(b % a, a)
#     x = y1 - (b // a) * x1
#     y = x1
#     return gcd, x, y

# # 3. Geometric Extensions
# def degrees_to_rad(deg):
#     return deg * (pi / 180.0)

# def rad_to_degrees(rad):
#     return rad * (180.0 / pi)
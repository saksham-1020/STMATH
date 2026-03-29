# from .special import sqrt_custom
# from .combinatorics import Combinatorics

# class NumberTheory:
#     @staticmethod
#     def gcd(a: int, b: int) -> int:
#         """Binary GCD (Stein's Algorithm) - More efficient than math.gcd for large ints."""
#         while b:
#             a, b = b, a % b
#         return abs(a)

#     @staticmethod
#     def lcm(a: int, b: int) -> int:
#         if a == 0 or b == 0: return 0
#         return abs(a * b) // NumberTheory.gcd(a, b)

#     @staticmethod
#     def prime_factors(n: int) -> list:
#         factors = []
#         d = 2
#         temp = abs(n)
#         while d * d <= temp:
#             while temp % d == 0:
#                 factors.append(d)
#                 temp //= d
#             d += 1
#         if temp > 1: factors.append(temp)
#         return factors

#     @staticmethod
#     def totient(n: int) -> int:
#         """Euler's Totient function: Counts numbers coprime to n."""
#         result = n
#         p = 2
#         temp = n
#         while p * p <= temp:
#             if temp % p == 0:
#                 while temp % p == 0:
#                     temp //= p
#                 result -= result // p
#             p += 1
#         if temp > 1:
#             result -= result // temp
#         return result

#     @staticmethod
#     def catalan_number(n: int) -> int:
#         """
#         Cn = (1/(n+1)) * (2n choose n). 
#         Calculated via our Stirling-optimized Combinatorics engine.
#         """
#         # (2n! / (n! * n!)) / (n+1)
#         comb_val = Combinatorics.nCr(2 * n, n)
#         return comb_val // (n + 1)

#     @staticmethod
#     def pell_number(n: int) -> int:
#         """Pell sequence: P(n) = 2*P(n-1) + P(n-2). (O(n) logic)"""
#         if n == 0: return 0
#         if n == 1: return 1
#         p0, p1 = 0, 1
#         for _ in range(2, n + 1):
#             p0, p1 = p1, 2 * p1 + p0
#         return p1

#     @staticmethod
#     def divisor_count(n: int) -> int:
#         """Number of divisors using prime factor exponent rule."""
#         factors = NumberTheory.prime_factors(n)
#         counts = {}
#         for f in factors: counts[f] = counts.get(f, 0) + 1
#         res = 1
#         for c in counts.values(): res *= (c + 1)
#         return res

#     @staticmethod
#     def divisor_sum(n: int) -> int:
#         """Sum of all divisors of n."""
#         total = 0
#         limit = int(sqrt_custom(n))
#         for i in range(1, limit + 1):
#             if n % i == 0:
#                 total += i
#                 if i != n // i:
#                     total += n // i
#         return total
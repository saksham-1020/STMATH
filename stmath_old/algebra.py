# import cmath
# from typing import Tuple, List

# def solve_linear(a: float, b: float) -> float:
#     if a == 0:
#         raise ZeroDivisionError("a cannot be zero.")
#     return -b / a

# def quadratic_roots(a: float, b: float, c: float) -> Tuple[complex, complex]:
#     if a == 0:
#         raise ZeroDivisionError("a cannot be zero.")
#     D = b**2 - 4 * a * c
#     sqrtD = cmath.sqrt(D)
#     return (-b + sqrtD) / (2 * a), (-b - sqrtD) / (2 * a)

# class LinearSystem:
#     @staticmethod
#     def solve(A: List[List[float]], b: List[float]) -> List[float]:
#         n = len(A)
#         for i in range(n):
#             pivot = A[i][i]
#             if abs(pivot) < 1e-12:
#                 for k in range(i + 1, n):
#                     if abs(A[k][i]) > abs(pivot):
#                         A[i], A[k] = A[k], A[i]
#                         b[i], b[k] = b[k], b[i]
#                         pivot = A[i][i]
#                         break
#             if abs(pivot) < 1e-12: raise ValueError("Matrix is singular")
#             for j in range(i, n): A[i][j] /= pivot
#             b[i] /= pivot
#             for k in range(n):
#                 if k != i:
#                     factor = A[k][i]
#                     for j in range(i, n): A[k][j] -= factor * A[i][j]
#                     b[k] -= factor * b[i]
#         return b

#     @staticmethod
#     def matrix_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
#         rows_A, cols_A = len(A), len(A[0])
#         rows_B, cols_B = len(B), len(B[0])
#         if cols_A != rows_B: raise ValueError("Dimension mismatch")
#         res = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
#         for i in range(rows_A):
#             for j in range(cols_B):
#                 for k in range(cols_A):
#                     res[i][j] += A[i][k] * B[k][j]
#         return res
    
# class LinearAlgebra:
#     @staticmethod
#     def solve_system(A, b):
#         n = len(A)
#         for i in range(n):
#             max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
#             A[i], A[max_row] = A[max_row], A[i]
#             b[i], b[max_row] = b[max_row], b[i]
#             pivot = A[i][i]
#             if abs(pivot) < 1e-15: raise ValueError("Singular matrix")
#             for j in range(i, n): A[i][j] /= pivot
#             b[i] /= pivot
#             for k in range(n):
#                 if k != i:
#                     factor = A[k][i]
#                     for j in range(i, n): A[k][j] -= factor * A[i][j]
#                     b[k] -= factor * b[i]
#         return b
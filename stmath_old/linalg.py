# class LinearAlgebra:
#     @staticmethod
#     def lu_decomposition(A):
#         n = len(A)
#         L = [[0.0] * n for _ in range(n)]
#         U = [[0.0] * n for _ in range(n)]
#         for i in range(n):
#             L[i][i] = 1.0
#             for j in range(i, n):
#                 U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
#             for j in range(i + 1, n):
#                 L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
#         return L, U

#     @staticmethod
#     def determinant(A):
#         n = len(A)
#         if n == 1: return A[0][0]
#         if n == 2: return A[0][0]*A[1][1] - A[0][1]*A[1][0]
#         det = 0
#         for c in range(n):
#             minor = [row[:c] + row[c+1:] for row in A[1:]]
#             det += ((-1)**c) * A[0][c] * LinearAlgebra.determinant(minor)
#         return det

#     @staticmethod
#     def transpose(A):
#         return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
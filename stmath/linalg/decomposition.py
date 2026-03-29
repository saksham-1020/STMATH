class Decomposition:

    @staticmethod
    def lu(A):
        n = len(A)
        L = [[0]*n for _ in range(n)]
        U = [[0]*n for _ in range(n)]

        for i in range(n):
            L[i][i] = 1

            for j in range(i, n):
                U[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))

            for j in range(i+1, n):
                L[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]

        return L, U
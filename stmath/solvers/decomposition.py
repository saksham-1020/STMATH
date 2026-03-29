class LU:

    @staticmethod
    def decompose(A):
        n = len(A)
        L = [[0]*n for _ in range(n)]
        U = [[0]*n for _ in range(n)]

        for i in range(n):
            L[i][i] = 1

            for j in range(i, n):
                U[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))

            for j in range(i+1, n):
                if U[i][i] == 0:
                    raise ValueError("Singular matrix")
                L[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]

        return L, U

    @staticmethod
    def solve(A, b):
        L, U = LU.decompose(A)
        n = len(A)

        # Forward substitution (Ly = b)
        y = [0]*n
        for i in range(n):
            y[i] = b[i] - sum(L[i][j]*y[j] for j in range(i))

        # Backward substitution (Ux = y)
        x = [0]*n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]

        return x
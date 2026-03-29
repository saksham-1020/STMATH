class LinearAlgebra:

    @staticmethod
    def matmul(A, B):
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])

        if cols_A != rows_B:
            raise ValueError("Dimension mismatch")

        return [
            [sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)]
            for i in range(rows_A)
        ]

    @staticmethod
    def transpose(A):
        return list(map(list, zip(*A)))

    @staticmethod
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    @staticmethod
    def norm(v):
        return sum(x*x for x in v) ** 0.5

    @staticmethod
    def identity(n):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    @staticmethod
    def determinant(A):
        if len(A) == 1:
            return A[0][0]
        if len(A) == 2:
            return A[0][0]*A[1][1] - A[0][1]*A[1][0]

        det = 0
        for c in range(len(A)):
            minor = [row[:c] + row[c+1:] for row in A[1:]]
            det += ((-1)**c) * A[0][c] * LinearAlgebra.determinant(minor)
        return det

    @staticmethod
    def inverse(A):
        n = len(A)
        I = LinearAlgebra.identity(n)

        # Augmented matrix
        aug = [A[i] + I[i] for i in range(n)]

        # Gauss-Jordan
        for i in range(n):
            pivot = aug[i][i]
            if abs(pivot) < 1e-12:
                raise ValueError("Singular matrix")

            for j in range(2*n):
                aug[i][j] /= pivot

            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2*n):
                        aug[k][j] -= factor * aug[i][j]

        return [row[n:] for row in aug]

    @staticmethod
    def solve(A, b):
        n = len(A)

        for i in range(n):
            max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]

            pivot = A[i][i]
            if abs(pivot) < 1e-12:
                raise ValueError("Singular matrix")

            for j in range(i, n):
                A[i][j] /= pivot
            b[i] /= pivot

            for k in range(n):
                if k != i:
                    factor = A[k][i]
                    for j in range(i, n):
                        A[k][j] -= factor * A[i][j]
                    b[k] -= factor * b[i]

        return b
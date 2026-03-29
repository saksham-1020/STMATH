class QR:

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
    def decompose(A):
        n = len(A)
        m = len(A[0])

        Q = [[0]*m for _ in range(n)]
        R = [[0]*m for _ in range(m)]

        A_T = QR.transpose(A)

        U = []

        for i in range(m):
            u = A_T[i][:]

            for j in range(i):
                proj = QR.dot(A_T[i], U[j]) / QR.dot(U[j], U[j])
                u = [ui - proj*uj for ui, uj in zip(u, U[j])]

            U.append(u)

            norm_u = QR.norm(u)
            for k in range(n):
                Q[k][i] = u[k] / norm_u

            for j in range(i, m):
                R[i][j] = QR.dot(QR.transpose(Q)[i], A_T[j])

        return Q, R
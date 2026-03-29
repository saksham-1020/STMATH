class CG:

    @staticmethod
    def matvec(A, x):
        return [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]

    @staticmethod
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    @staticmethod
    def solve(A, b, iters=100):
        n = len(b)
        x = [0]*n
        r = [b[i] - CG.matvec(A, x)[i] for i in range(n)]
        p = r[:]

        for _ in range(iters):
            Ap = CG.matvec(A, p)
            alpha = CG.dot(r, r) / CG.dot(p, Ap)

            x = [x[i] + alpha*p[i] for i in range(n)]
            r_new = [r[i] - alpha*Ap[i] for i in range(n)]

            if sum(abs(v) for v in r_new) < 1e-6:
                break

            beta = CG.dot(r_new, r_new) / CG.dot(r, r)
            p = [r_new[i] + beta*p[i] for i in range(n)]
            r = r_new

        return x
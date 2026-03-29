class Eigen:

    @staticmethod
    def power_iteration(A, iterations=100):
        n = len(A)
        b = [1.0]*n

        for _ in range(iterations):
            b_new = [sum(A[i][j]*b[j] for j in range(n)) for i in range(n)]
            norm = sum(x*x for x in b_new)**0.5
            b = [x/norm for x in b_new]

        return b
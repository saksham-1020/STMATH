class GradientDescent:

    @staticmethod
    def matvec(A, x):
        return [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]

    @staticmethod
    def solve(A, b, lr=0.001, epochs=100):
        n = len(b)
        x = [0]*n

        for _ in range(epochs):
            Ax = GradientDescent.matvec(A, x)
            grad = [Ax[i] - b[i] for i in range(n)]
            x = [x[i] - lr*grad[i] for i in range(n)]

        return x
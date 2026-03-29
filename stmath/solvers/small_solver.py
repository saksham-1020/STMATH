from .decomposition import LU
from .regularized import RegularizedSolver

class SmallSolver:

    def solve(self, X, y, regularization=None):

        XT = list(map(list, zip(*X)))

        XTX = [
            [sum(XT[i][k]*X[k][j] for k in range(len(X))) for j in range(len(X[0]))]
            for i in range(len(X[0]))
        ]

        XTy = [
            sum(XT[i][k]*y[k] for k in range(len(X)))
            for i in range(len(X[0]))
        ]

        if regularization == "ridge":
            return RegularizedSolver.ridge(XTX, XTy)

        return LU.solve(XTX, XTy)
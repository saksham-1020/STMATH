from .decomposition import LU

class RegularizedSolver:

    @staticmethod
    def ridge(XTX, XTy, alpha=0.1):
        n = len(XTX)

        for i in range(n):
            XTX[i][i] += alpha

        return LU.solve(XTX, XTy)
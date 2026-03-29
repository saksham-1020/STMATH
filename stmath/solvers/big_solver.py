class BigSolver:

    def __init__(self, lr=0.001):
        self.lr = lr
        self.w = None

    def fit(self, X, y, epochs=1):

        n_features = len(X[0])
        self.w = [0]*n_features

        for _ in range(epochs):
            for i in range(len(X)):
                xi = X[i]
                yi = y[i]

                pred = sum(self.w[j]*xi[j] for j in range(n_features))
                error = pred - yi

                for j in range(n_features):
                    self.w[j] -= self.lr * error * xi[j]

        return self.w
from ..core.value import Value
from ..core.math_kernels import sqrt

class LinearRegression:
    def __init__(self, n_features=None):
        self.n_features = n_features
        self.w = None
        self.b = Value(0.0)

    def linear(self, x):
        out = Value(0.0)
        for wi , xi in zip(self.w, x):
            out = out + (wi * xi)
        return out + self.b

    def predict(self, X):
        if isinstance(X,(int, float)):
            X = [[X]]

        elif isinstance(X, list) and not isinstance(X[0], list):
            X=[X]

        preds = []

        for x in X:
            y_pred = self.linear(x)
            preds.append(y_pred.data)

        return preds[0] if len(preds) == 1 else preds
    
    def fit(self, X, y, lr=0.01, epochs=100):

        # ✅ handle 1D input
        if not isinstance(X[0], list):
            X = [[xi] for xi in X]

        # ✅ auto init weights
        if self.w is None:
            self.n_features = len(X[0])
            self.w = [Value(0.0) for _ in range(self.n_features)]

        for epoch in range(epochs):
            total_loss = 0

            for xi, yi in zip(X, y):

                # forward
                y_pred = self.linear(xi)

                # loss (MSE)
                loss = (y_pred.data - yi) ** 2

                # reset grads
                for wi in self.w:
                    wi.grad = 0.0
                self.b.grad = 0.0

                # gradient
                error = y_pred.data - yi

                for j in range(len(self.w)):
                    self.w[j].grad += 2 * error * xi[j]

                self.b.grad += 2 * error

                # update
                for wi in self.w:
                    wi.data -= lr * wi.grad

                self.b.data -= lr * self.b.grad

                total_loss += loss

            print(f"Epoch {epoch} Loss: {total_loss}")

    def score(self, X, y):
        preds = self.predict(X)
        mse = sum((pi - yi) ** 2 for pi, yi in zip(preds, y)) / len(y)
        return mse

    def rmse(self, X, y):
        return sqrt(self.score(X, y))
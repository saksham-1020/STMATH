class Pipeline:
    def __init__(self, steps):
        self.steps = steps  # list of (name, object)

    def fit(self, X, y):
        data = X

        for name, step in self.steps:
            if hasattr(step, "fit"):
                step.fit(data, y)

        return self

    def predict(self, X):
        data = X

        for name, step in self.steps:
            if hasattr(step, "predict"):
                data = step.predict(data)

        return data

    def score(self, X, y):
        for name, step in self.steps:
            if hasattr(step, "score"):
                return step.score(X, y)
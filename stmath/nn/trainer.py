class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, X, y, lr=0.01, epochs=50):
        for epoch in range(epochs):
            total_loss = 0

            for xi, yi in zip(X, y):
                pred = self.model(xi)
                loss = (pred - yi)**2

                for p in self.model.parameters():
                    p.grad = 0

                loss.backward()

                for p in self.model.parameters():
                    p.data -= lr * p.grad

                total_loss += loss.data

            print(f"Epoch {epoch}, Loss: {total_loss}")
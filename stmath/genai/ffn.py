import math

class FFN:

    @staticmethod
    def relu(x):
        return [max(0, xi) for xi in x]

    @staticmethod
    def forward(x):
        # simple linear + activation
        return [FFN.relu(row) for row in x]
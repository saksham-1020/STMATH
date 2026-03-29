# import math

# class Functional:
#     @staticmethod
#     def sigmoid(x):
#         return 1.0 / (1.0 + math.exp(-x))

#     @staticmethod
#     def relu(x):
#         return max(0.0, x)

#     @staticmethod
#     def tanh(x):
#         return math.tanh(x)

#     @staticmethod
#     def softmax(xs):
#         max_x = max(xs)
#         exps = [math.exp(x - max_x) for x in xs]
#         sum_exps = sum(exps)
#         return [e / sum_exps for e in exps]

#     @staticmethod
#     def mse_loss(y_true, y_pred):
#         return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)

#     @staticmethod
#     def cross_entropy(y_true, y_pred):
#         eps = 1e-15
#         return -sum(t * math.log(max(p, eps)) for t, p in zip(y_true, y_pred))
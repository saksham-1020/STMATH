# import math
# from typing import Sequence, List
# from .special import exp_custom, fast_ln, sqrt_custom

# class Activation:
#     @staticmethod
#     def sigmoid(x: float) -> float:
#         # Using custom exp to avoid math wrapper
#         return 1.0 / (1.0 + exp_custom(-x))

#     @staticmethod
#     def relu(x: float) -> float:
#         return x if x > 0 else 0.0

#     @staticmethod
#     def leaky_relu(x: float, alpha: float = 0.01) -> float:
#         # Advanced: Prevents 'Dying ReLU' problem
#         return x if x > 0 else alpha * x

#     @staticmethod
#     def elu(x: float, alpha: float = 1.0) -> float:
#         # Exponential Linear Unit for faster convergence
#         return x if x > 0 else alpha * (exp_custom(x) - 1)

#     @staticmethod
#     def tanh(x: float) -> float:
#         e_pos = exp_custom(x)
#         e_neg = exp_custom(-x)
#         return (e_pos - e_neg) / (e_pos + e_neg)

#     @staticmethod
#     def softmax(xs: Sequence[float]) -> List[float]:
#         if not xs: return []
#         max_x = max(xs)
#         # Using custom exp and numerical stability shift
#         exps = [exp_custom(x - max_x) for x in xs]
#         s = sum(exps)
#         return [e / s for e in exps]

# class Loss:
#     @staticmethod
#     def binary_cross_entropy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
#         eps = 1e-15
#         total_loss = 0.0
#         ln2 = fast_ln(2.0)
#         for t, p in zip(y_true, y_pred):
#             # Clipping for stability
#             p = max(eps, min(1 - eps, p))
#             # H(p, q) = -(y log(p) + (1-y) log(1-p))
#             total_loss += -(t * fast_ln(p) + (1 - t) * fast_ln(1 - p))
#         return total_loss / len(y_true)

#     @staticmethod
#     def categorical_cross_entropy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
#         # Professional: Used in Multi-class classification (Google Vision level)
#         eps = 1e-15
#         total_loss = 0.0
#         for t, p in zip(y_true, y_pred):
#             p = max(eps, min(1 - eps, p))
#             total_loss -= t * fast_ln(p)
#         return total_loss

#     @staticmethod
#     def huber_loss(y_true: float, y_pred: float, delta: float = 1.0) -> float:
#         # Advanced: Robust to outliers (Used in Finance/Robotics)
#         error = abs(y_true - y_pred)
#         if error <= delta:
#             return 0.5 * (error ** 2)
#         else:
#             return delta * (error - 0.5 * delta)

# class Optimization:
#     @staticmethod
#     def dropout(x: List[float], rate: float = 0.5) -> List[float]:
#         # Simple simulation of neural network dropout
#         import random
#         return [val if random.random() > rate else 0.0 for val in x]
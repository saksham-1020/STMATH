# from typing import Sequence, List, Dict, Tuple, Any
# from .special import sqrt_custom, fast_ln

# class RegressionMetrics:
#     @staticmethod
#     def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
#         n = len(y_true)
#         if n == 0: return 0.0
#         return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n

#     @staticmethod
#     def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
#         # Zero-dependency RMSE using our custom Newton-Raphson kernel
#         return sqrt_custom(RegressionMetrics.mse(y_true, y_pred))

#     @staticmethod
#     def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
#         n = len(y_true)
#         return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n if n > 0 else 0.0

#     @staticmethod
#     def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
#         """Mean Absolute Percentage Error: Standard for Finance/Sales forecasting."""
#         n = len(y_true)
#         if n == 0: return 0.0
#         # eps added to avoid division by zero
#         return (sum(abs((a - b) / a) if a != 0 else abs(b) for a, b in zip(y_true, y_pred)) / n) * 100

#     @staticmethod
#     def r2_score(y_true: Sequence[float], y_pred: Sequence[float], p: int = 0) -> float:
#         """Computes R2 or Adjusted R2 if 'p' (number of predictors) is provided."""
#         n = len(y_true)
#         if n <= 1: return 0.0
#         mean_y = sum(y_true) / n
#         ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
#         ss_tot = sum((a - mean_y) ** 2 for a in y_true)
#         r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
#         if p > 0 and n > p + 1:
#             return 1 - (1 - r2) * (n - 1) / (n - p - 1)
#         return r2

# class ClassificationMetrics:
#     @staticmethod
#     def confusion_matrix(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, Dict[str, int]]:
#         """Multi-class Confusion Matrix implementation from scratch."""
#         matrix = {}
#         classes = sorted(list(set(y_true) | set(y_pred)))
#         for true_cls in classes:
#             matrix[true_cls] = {pred_cls: 0 for pred_cls in classes}
        
#         for t, p in zip(y_true, y_pred):
#             matrix[t][p] += 1
#         return matrix

#     @staticmethod
#     def binary_report(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
#         """Precision, Recall, F1, and Accuracy in one go for Binary Tasks."""
#         tp = fp = tn = fn = 0
#         for t, p in zip(y_true, y_pred):
#             if t == 1 and p == 1: tp += 1
#             elif t == 0 and p == 1: fp += 1
#             elif t == 0 and p == 0: tn += 1
#             elif t == 1 and p == 0: fn += 1
            
#         prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
#         acc = (tp + tn) / len(y_true) if y_true else 0.0
        
#         return {"precision": prec, "recall": rec, "f1_score": f1, "accuracy": acc}

#     @staticmethod
#     def log_loss(y_true: Sequence[int], y_probs: Sequence[float]) -> float:
#         """Binary Cross-Entropy Loss with Numerical Clipping Stability."""
#         eps = 1e-15
#         loss = 0.0
#         for t, p in zip(y_true, y_probs):
#             p = max(eps, min(1 - eps, p))
#             loss += -(t * fast_ln(p) + (1 - t) * fast_ln(1 - p))
#         return loss / len(y_true) if y_true else 0.0

#     @staticmethod
#     def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
#         """Balanced Accuracy: Crucial for imbalanced datasets (MNC Grade)."""
#         report = ClassificationMetrics.binary_report(y_true, y_pred)
#         # Specificity = TN / (TN + FP)
#         tp, fp, tn, fn = 0, 0, 0, 0
#         for t, p in zip(y_true, y_pred):
#             if t == 1 and p == 1: tp += 1
#             elif t == 0 and p == 1: fp += 1
#             elif t == 0 and p == 0: tn += 1
#             elif t == 1 and p == 0: fn += 1
        
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#         return (sensitivity + specificity) / 2
# # \"\"\"GenAI math helpers: logits->prob, softmax temperature, simple attention scores\"\"\"
# import math
# from typing import List


# def logits_to_prob(logits: List[float]) -> List[float]:
#     ex = [math.exp(x) for x in logits]
#     s = sum(ex) or 1.0
#     return [x / s for x in ex]


# def softmax_temperature(logits: List[float], T: float = 1.0):
#     if T <= 0:
#         raise ValueError("Temperature must be > 0")
#     ex = [math.exp(x / T) for x in logits]
#     s = sum(ex) or 1.0
#     return [x / s for x in ex]


# def attention_scores(queries, keys, softmax=True):
#     # queries, keys: list-of-lists (vector per token). returns attention matrix normalized row-wise.
#     import math

#     # compute scaled dot product
#     scores = []
#     for q in queries:
#         row = []
#         for k in keys:
#             # dot product (ensure numeric)
#             dot = sum(a * b for a, b in zip(q, k))
#             row.append(dot)
#         scores.append(row)
#     if softmax:

#         def softmax_row(r):
#             ex = [math.exp(x) for x in r]
#             s = sum(ex) or 1.0
#             return [x / s for x in ex]

#         return [softmax_row(r) for r in scores]
#     return scores

# import math
# from typing import List
# from .special import exp_custom, sqrt_custom

# class GenAIEngine:
#     @staticmethod
#     def softmax_with_temp(logits: List[float], T: float = 1.0) -> List[float]:
#         """
#         Softmax with Temperature scaling and Max-Subtraction for Numerical Stability.
#         Formula: exp((x - max) / T) / sum(exp((x - max) / T))
#         """
#         if T <= 0: raise ValueError("Temperature must be positive")
        
#         # Shift logits by max to prevent exp(large_number) overflow
#         max_l = max(logits)
#         scaled_shifted = [(x - max_l) / T for x in logits]
        
#         exps = [exp_custom(x) for x in scaled_shifted]
#         s = sum(exps)
#         return [e / (s if s != 0 else 1.0) for e in exps]

#     @staticmethod
#     def scaled_dot_product_attention(queries: List[List[float]], 
#                                     keys: List[List[float]], 
#                                     values: List[List[float]]) -> List[List[float]]:
#         """
#         Core Transformer Attention: softmax(QK^T / sqrt(d_k))V
#         Used in GPT, Llama, and Gemini architectures.
#         """
#         d_k = len(keys[0])
#         # Scaling factor to keep gradients stable
#         scale = sqrt_custom(d_k)
        
#         # 1. MatMul: Q * K^T and Scale
#         scores = []
#         for q in queries:
#             row = []
#             for k in keys:
#                 dot = sum(qi * ki for qi, ki in zip(q, k))
#                 row.append(dot / scale)
#             scores.append(row)
            
#         # 2. Softmax along rows (Attention Weights)
#         weights = [GenAIEngine.softmax_with_temp(row) for row in scores]
        
#         # 3. MatMul: Weights * V (Weighted Sum)
#         output = []
#         for w_row in weights:
#             new_row = [0.0] * len(values[0])
#             for i, weight in enumerate(w_row):
#                 for j, v in enumerate(values[i]):
#                     new_row[j] += weight * v
#             output.append(new_row)
            
#         return output

#     @staticmethod
#     def layer_norm(x: List[float]):
#         """Layer Normalization: Essential for deep transformer stability."""
#         mean = sum(x) / len(x)
#         variance = sum((xi - mean)**2 for xi in x) / len(x)
#         std = sqrt_custom(variance + 1e-6)
#         return [(xi - mean) / std for xi in x]
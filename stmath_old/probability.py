# from typing import Iterable, List, Dict
# from .combinatorics import Combinatorics # Import from existing file
# from .special import sqrt_custom

# class ProbabilityEngine:
#     @staticmethod
#     def bayes_theorem(p_a: float, p_b_given_a: float, p_a_not: float, p_b_given_a_not: float) -> float:
#         """
#         Advanced Bayesian Inference: P(A|B).
#         Uses Law of Total Probability for the denominator.
#         """
#         numerator = p_b_given_a * p_a
#         denominator = (p_b_given_a * p_a) + (p_b_given_a_not * p_a_not)
#         if denominator == 0: raise ZeroDivisionError("P(B) cannot be zero")
#         return numerator / denominator

#     @staticmethod
#     def expected_value(values: Iterable[float], probs: Iterable[float]) -> float:
#         """E[X] = Σ x * p(x). Includes MNC-grade normalization check."""
#         v_list, p_list = list(values), list(probs)
#         if len(v_list) != len(p_list): raise ValueError("Length mismatch")
        
#         total_p = sum(p_list)
#         if abs(total_p - 1.0) > 1e-6:
#             # Auto-normalization if data is slightly off
#             p_list = [p / total_p for p in p_list]
            
#         return sum(v * p for v, p in zip(v_list, p_list))

#     @staticmethod
#     def variance_discrete(values: Iterable[float], probs: Iterable[float]) -> float:
#         """Var(X) = E[X^2] - (E[X])^2. Foundation for risk modeling."""
#         v_list, p_list = list(values), list(probs)
#         ev = ProbabilityEngine.expected_value(v_list, p_list)
#         ev_sq = sum((v**2) * p for v, p in zip(v_list, p_list))
#         return ev_sq - (ev**2)

#     @staticmethod
#     def conditional_prob(joint_p_xy: float, marginal_p_y: float) -> float:
#         """P(X|Y) = P(X∩Y) / P(Y)."""
#         return joint_p_xy / marginal_p_y if marginal_p_y != 0 else 0.0

#     @staticmethod
#     def is_independent(p_a: float, p_b: float, p_joint_ab: float) -> bool:
#         """Checks if two events are independent: P(A∩B) == P(A) * P(B)."""
#         return abs(p_joint_ab - (p_a * p_b)) < 1e-9
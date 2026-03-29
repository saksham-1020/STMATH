# import math

# def exp_custom(x, n=50):
#     res = 1.0
#     term = 1.0
#     for i in range(1, n):
#         term *= x / i
#         res += term
#     return res

# def sqrt_custom(x):
#     if x < 0: raise ValueError
#     if x == 0: return 0.0
#     last = 0.0
#     res = x
#     while res != last:
#         last = res
#         res = (res + x / res) / 2
#     return res

# def gamma_pro(x):
#     if x == 1.0: return 1.0
#     if x < 0.5: return math.pi / (math.sin(math.pi * x) * gamma_pro(1 - x))
#     x -= 1
#     p = [676.5203681218851, -1259.1392167224028, 771.32342877765313, 
#          -176.61502916214059, 12.507343278686905, -0.13857109526572012, 
#          9.9843695780195716e-6, 1.5056327351493116e-7]
#     g = 7
#     res = 0.99999999999980993
#     for i, range_p in enumerate(p):
#         res += range_p / (x + i + 1)
#     t = x + g + 0.5
#     return sqrt_custom(2 * math.pi) * t**(x + 0.5) * exp_custom(-t) * res

# def erf_pro(x):
#     # Winitzki approximation (Max error < 0.0001)
#     a = (8 * (math.pi - 3)) / (3 * math.pi * (4 - math.pi))
#     x2 = x * x
#     inner = exp_custom(-x2 * (4/math.pi + a*x2) / (1 + a*x2))
#     return math.copysign(sqrt_custom(1 - inner), x)

# def fast_exp(x, n=100):
#     res, term = 1.0, 1.0
#     for i in range(1, n):
#         term *= x / i
#         res += term
#     return res

# def fast_sqrt(x):
#     if x < 0: raise ValueError
#     curr = x
#     while True:
#         prev = curr
#         curr = (curr + x / curr) / 2
#         if abs(curr - prev) < 1e-15: break
#     return curr

# def fast_ln(x):
#     if x <= 0: raise ValueError
#     res = 0.0
#     for _ in range(100):
#         e_res = fast_exp(res)
#         res += 2 * ((x - e_res) / (x + e_res))
#     return res
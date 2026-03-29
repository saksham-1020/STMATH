# """
# AIMATHX (STMATH) — The Zero-Dependency Numerical Ecosystem.
# Built for Research, AI, Blockchain, and Quantum Computing.
# Author: Saksham Tomar (saksham-1020)
# """

# # =====================================================
# # CORE & SPECIAL KERNELS (The Foundation)
# # =====================================================
# from .core import add, sub, mul, div, square, cube, sqrt, power, percent, percent_change
# from .special import exp_custom as exp, fast_ln as ln, sqrt_custom, erf_pro, gamma_pro

# # =====================================================
# # SCIENTIFIC & TRANSCENDENTAL
# # =====================================================
# from .scientific import Scientific
# sin, cos, tan = Scientific.sin, Scientific.cos, Scientific.tan
# factorial, deg2rad, rad2deg = Scientific.factorial, Scientific.deg2rad, Scientific.rad2deg
# log10 = Scientific.log10

# # =====================================================
# # STATISTICS & PROBABILITY
# # =====================================================
# from .statistics import Statistics
# mean, median, mode = Statistics.mean, Statistics.median, Statistics.mode
# variance, std, z_score = Statistics.variance, Statistics.std_dev, Statistics.z_score
# iqr, skewness, correlation = Statistics.iqr, Statistics.skewness, Statistics.correlation

# from .probability import ProbabilityEngine
# bayes, expected_value = ProbabilityEngine.bayes_theorem, ProbabilityEngine.expected_value
# from .combinatorics import Combinatorics
# nCr, nPr = Combinatorics.nCr, Combinatorics.nPr

# from .distributions import (
#     normal_pdf, normal_cdf, bernoulli_pmf, binomial_pmf, 
#     poisson_pmf, exponential_pdf, uniform_pdf, t_pdf, chi_square_pdf
# )

# # =====================================================
# # MACHINE LEARNING & DEEP LEARNING
# # =====================================================
# from .ml import RegressionMetrics, ClassificationMetrics
# mse, rmse, mae, r2_score = RegressionMetrics.mse, RegressionMetrics.rmse, RegressionMetrics.mae, RegressionMetrics.r2_score
# log_loss, confusion_matrix = ClassificationMetrics.log_loss, ClassificationMetrics.confusion_matrix

# from .dl import Activation, Loss
# sigmoid, relu, tanh, softmax = Activation.sigmoid, Activation.relu, Activation.tanh, Activation.softmax
# binary_cross_entropy = Loss.binary_cross_entropy

# # =====================================================
# # NLP & GEN-AI
# # =====================================================
# from .nlp import Tokenizer, Vectorizer, NLPSimilarity
# tfidf = Vectorizer.tf_idf
# cosine_similarity = NLPSimilarity.cosine_similarity

# from .genai import GenAIEngine
# attention_scores = GenAIEngine.scaled_dot_product_attention
# softmax_temperature = GenAIEngine.softmax_with_temp

# # =====================================================
# # GRAPH & PATHFINDING (Phase 2)
# # =====================================================
# from .graph import GraphEngine, Visualizer
# dijkstra = GraphEngine.dijkstra
# a_star = GraphEngine.a_star
# pagerank = GraphEngine.pagerank
# bfs, dfs = GraphEngine.bfs, GraphEngine.dfs
# draw_graph = Visualizer.draw_graph

# # =====================================================
# # CRYPTO & BLOCKCHAIN
# # =====================================================
# from .crypto_math import SHA256, CryptoMath
# sha256 = SHA256.hash
# merkle_root = CryptoMath.merkle_root

# # =====================================================
# # FINANCE & TIME SERIES
# # =====================================================
# from .finance_math import FinanceMath, RiskAnalysis
# cagr, npv = FinanceMath.cagr, FinanceMath.net_present_value
# sharpe_ratio = RiskAnalysis.sharpe_ratio

# from .timeseries import TimeSeries
# sma, ema, rsi = TimeSeries.sma, TimeSeries.ema, TimeSeries.rsi

# # =====================================================
# # NUMBER THEORY
# # =====================================================
# from .number_theory import NumberTheory
# gcd, lcm, is_prime = NumberTheory.gcd, NumberTheory.lcm, NumberTheory.divisor_count
# totient, catalan = NumberTheory.totient, NumberTheory.catalan_number

# # =====================================================
# # BENCHMARK & UTILS
# # =====================================================
# from .utils import Utils
# timeit = Utils.time_it
# flatten = Utils.flatten

# __version__ = "2.0.0"
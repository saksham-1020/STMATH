# =========================================================
# STMATH: Unified AI + Math Engine (Zero Dependency Core)
# =========================================================

# ==============================
# CORE
# ==============================
from .core.value import Value
from .core.tensor import Tensor

from .core.ops import (
    add, sub, mul, div,
    square
)

from .core.math_kernels import (
    sqrt, exp, log,
    relu, tanh
)
# ==============================
# STATISTICS
# ==============================
from .stats.statistics import Statistics
# ==============================
# LINEAR ALGEBRA
# ==============================
from .linalg.linalg import LinearAlgebra
# ==============================
# GRAPH
# ==============================
from .graph.graph import GraphEngine, GraphPipeline
from . import graph
# ==============================
# MACHINE LEARNING
# ==============================
from .ml.linear_regression import LinearRegression
from .ml.logistic_regression import LogisticRegression
from .ml.metrics import Metrics
# ==============================
# DEEP LEARNING (NN)
# ==============================
from .nn.layers import MLP
from .nn.trainer import Trainer
from .nn.functional import Functional
from .nn.models import simple_mlp
# ==============================
# NLP
# ==============================
from .nlp.vectorizer import Vectorizer
from .nlp.similarity import Similarity
# ==============================
# COMPUTER VISION
# ==============================
from .vision.vision import edge, convolve2d
# ==============================
# GENAI
# ==============================
from .genai.transformer import TransformerBlock
from .genai.pipeline import GenAIPipeline
# ==============================
# BECHNAMRK
# ==============================
from .utils.benchmark import Benchmark
# ==============================
# ENGINE (🔥 MAIN BRAIN)
# ==============================
from .engine.adaptive import AdaptiveSolver
# ==============================
# UTILITIES
# ==============================
from .core.gradcheck import grad_check
from .core.graph_utils import trace_graph
# ==============================
# INTERNAL MODULE ACCESS (ADVANCED USERS)
# ==============================
from . import engine, solvers
# ==============================
# HIGH-LEVEL API (🔥 MOST IMPORTANT)
# ==============================
def solve(X, y, **kwargs):
    """
    High-level unified solver (auto-adaptive)

    Example:
        import stmath as sm
        sm.solve(X, y)
    """
    return AdaptiveSolver().solve(X, y, **kwargs)
# ==============================
# EXPORT CONTROL
# ==============================
__all__ = [

    # ===== CORE =====
    "Value", "Tensor",
    "add", "sub", "mul", "div",
    "square",
    "sqrt", "exp", "log",
    "relu", "tanh",

    # ===== STATS =====
    "Statistics",

    # ===== LINALG =====
    "LinearAlgebra",

    # ===== GRAPH =====
    "GraphEngine", "GraphPipeline", "graph",

    # ===== ML =====
    "LinearRegression", "LogisticRegression", "Metrics",

    # ===== NN =====
    "MLP", "Trainer", "Functional", "simple_mlp",

    # ===== NLP =====
    "Vectorizer", "Similarity",

    # ===== VISION =====
    "edge", "convolve2d",

    # ===== GENAI =====
    "TransformerBlock", "GenAIPipeline",

    # ===== ENGINE =====
    "AdaptiveSolver",

    # ===== UTILITIES =====
    "grad_check", "trace_graph",

    # ==== BENCHMARK ====
    "Benchmark"

    # ===== MODULE ACCESS =====
    "engine", "solvers",

    # ===== HIGH LEVEL =====
    "solve"
]
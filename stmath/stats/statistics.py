from typing import List, Union, Optional
from collections import Counter
from stmath.core.math_kernels import sqrt
from stmath.utils.constants import EPS


class Statistics:

    @staticmethod
    def mean(data: List[float]) -> float:
        if not data:
            raise ValueError("Empty data")
        return sum(data) / len(data)

    @staticmethod
    def median(data: List[float]) -> float:
        n = len(data)
        if n == 0:
            raise ValueError("Empty data")

        s_data = sorted(data)
        mid = n // 2

        if n % 2 == 1:
            return s_data[mid]

        return (s_data[mid - 1] + s_data[mid]) / 2.0

    @staticmethod
    def mode(data: List[float]) -> Union[float, List[float], None]:
        if not data:
            raise ValueError("Empty data")

        counts = Counter(data)
        max_v = max(counts.values())

        if max_v == 1:
            return None

        modes = [k for k, v in counts.items() if v == max_v]
        return modes[0] if len(modes) == 1 else modes

    @staticmethod
    def variance(data: List[float], sample: bool = True) -> float:
        n = len(data)
        if n < 2:
            raise ValueError("Needs at least 2 points")

        mu = Statistics.mean(data)
        sse = sum((x - mu) ** 2 for x in data)

        return sse / (n - 1 if sample else n)

    @staticmethod
    def std_dev(data: List[float], sample: bool = True) -> float:
        return sqrt(Statistics.variance(data, sample))

    @staticmethod
    def z_score(x: float,
                data: Optional[List[float]] = None,
                mu: Optional[float] = None,
                sigma: Optional[float] = None) -> float:

        if data is not None:
            mu = Statistics.mean(data)
            sigma = Statistics.std_dev(data)

        if mu is None or sigma is None:
            raise ValueError("Provide either data or mean and std_dev")

        # 🔥 stability fix
        return (x - mu) / (sigma + EPS)

    @staticmethod
    def iqr(data: List[float]) -> float:
        n = len(data)

        if n < 4:
            raise ValueError("Needs at least 4 points")

        s_data = sorted(data)
        mid = n // 2

        low = s_data[:mid]
        high = s_data[mid + (1 if n % 2 != 0 else 0):]

        return Statistics.median(high) - Statistics.median(low)

    @staticmethod
    def skewness(data: List[float]) -> float:
        n = len(data)

        mu = Statistics.mean(data)
        sigma = Statistics.std_dev(data, sample=False)

        if sigma == 0:
            return 0.0

        m3 = sum((x - mu) ** 3 for x in data) / n
        return m3 / (sigma ** 3)

    @staticmethod
    def correlation(x: List[float], y: List[float]) -> float:

        if len(x) != len(y):
            raise ValueError("Length mismatch")

        mu_x = Statistics.mean(x)
        mu_y = Statistics.mean(y)

        num = sum((xi - mu_x) * (yi - mu_y)
                  for xi, yi in zip(x, y))

        den = sqrt(
            sum((xi - mu_x) ** 2 for xi in x) *
            sum((yi - mu_y) ** 2 for yi in y)
        )

        return num / (den + EPS)
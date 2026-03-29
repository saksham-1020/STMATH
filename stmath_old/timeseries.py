# from typing import Sequence, List, Optional
# from .special import sqrt_custom

# class TimeSeries:
#     @staticmethod
#     def sma(series: Sequence[float], window: int) -> List[Optional[float]]:
#         """Simple Moving Average with None padding for initial window."""
#         n = len(series)
#         if window <= 0: raise ValueError("Window must be positive")
        
#         result = [None] * (window - 1)
#         # Sliding window sum for performance
#         current_sum = sum(series[:window])
#         if n >= window:
#             result.append(current_sum / window)
#             for i in range(window, n):
#                 current_sum += series[i] - series[i - window]
#                 result.append(current_sum / window)
#         return result

#     @staticmethod
#     def ema(series: Sequence[float], span: int) -> List[float]:
#         """
#         Exponential Moving Average (EMA).
#         Alpha is calculated as 2 / (span + 1). (Industry Standard)
#         """
#         if not series: return []
#         alpha = 2 / (span + 1)
#         ema_vals = [series[0]]
#         for x in series[1:]:
#             ema_vals.append(alpha * x + (1 - alpha) * ema_vals[-1])
#         return ema_vals

#     @staticmethod
#     def bollinger_bands(series: Sequence[float], window: int = 20, k: float = 2):
#         """
#         Bollinger Bands: Middle (SMA), Upper (SMA + k*std), Lower (SMA - k*std).
#         Used to measure market volatility.
#         """
#         from .statistics import Statistics
#         sma_vals = TimeSeries.sma(series, window)
#         upper, lower = [], []
        
#         for i, m in enumerate(sma_vals):
#             if m is None:
#                 upper.append(None); lower.append(None)
#             else:
#                 window_data = series[i+1-window : i+1]
#                 std = Statistics.std_dev(window_data)
#                 upper.append(m + k * std)
#                 lower.append(m - k * std)
#         return sma_vals, upper, lower

#     @staticmethod
#     def rsi(series: Sequence[float], window: int = 14) -> List[Optional[float]]:
#         """
#         Relative Strength Index (RSI): Measures momentum (0 to 100).
#         RSI = 100 - (100 / (1 + RS))
#         """
#         n = len(series)
#         if n <= window: return [None] * n
        
#         deltas = [series[i] - series[i-1] for i in range(1, n)]
#         rsi_vals = [None] * (window)
        
#         # Initial average gain/loss
#         avg_gain = sum(max(d, 0) for d in deltas[:window]) / window
#         avg_loss = sum(abs(min(d, 0)) for d in deltas[:window]) / window
        
#         for i in range(window, n-1):
#             d = deltas[i]
#             gain = max(d, 0)
#             loss = abs(min(d, 0))
            
#             # Wilders Smoothing Method
#             avg_gain = (avg_gain * (window - 1) + gain) / window
#             avg_loss = (avg_loss * (window - 1) + loss) / window
            
#             if avg_loss == 0:
#                 rsi_vals.append(100.0)
#             else:
#                 rs = avg_gain / avg_loss
#                 rsi_vals.append(100 - (100 / (1 + rs)))
        
#         return rsi_vals
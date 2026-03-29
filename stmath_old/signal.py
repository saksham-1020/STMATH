# import math

# class Signal:
#     @staticmethod
#     def moving_average(data, window_size=3):
#         res = []
#         for i in range(len(data) - window_size + 1):
#             res.append(sum(data[i:i+window_size]) / window_size)
#         return res

#     @staticmethod
#     def dft(data):
#         # Discrete Fourier Transform - Pure Math
#         # No scipy.fft wrapper!
#         n = len(data)
#         frequencies = []
#         for k in range(n):
#             re, im = 0.0, 0.0
#             for t in range(n):
#                 angle = 2 * math.pi * k * t / n
#                 re += data[t] * math.cos(angle)
#                 im -= data[t] * math.sin(angle)
#             frequencies.append(complex(re, im))
#         return frequencies
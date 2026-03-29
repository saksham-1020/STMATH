# import math
# import random

# class Qubit:
#     def __init__(self):
#         self.state = [complex(1, 0), complex(0, 0)]

#     def apply_gate(self, gate_matrix):
#         a, b = self.state
#         self.state = [
#             gate_matrix[0][0] * a + gate_matrix[0][1] * b,
#             gate_matrix[1][0] * a + gate_matrix[1][1] * b
#         ]

#     def measure(self):
#         prob0 = abs(self.state[0])**2
#         outcome = 0 if random.random() < prob0 else 1
#         self.state = [complex(1, 0), complex(0, 0)] if outcome == 0 else [complex(0, 0), complex(1, 0)]
#         return outcome

# class QuantumGates:
#     H = [[1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), -1/math.sqrt(2)]]
#     X = [[0, 1], [1, 0]]
#     Y = [[0, complex(0, -1)], [complex(0, 1), 0]]
#     Z = [[1, 0], [0, -1]]

# def get_bloch_coordinates(qubit):
#     # Bonus advanced function for visualization
#     a, b = qubit.state
#     theta = 2 * math.acos(abs(a))
#     phi = math.atan2(b.imag, b.real) if abs(b) > 1e-9 else 0
#     return theta, phi
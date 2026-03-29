# import math
# from typing import List

# class SHA256:
#     """NIST FIPS 180-4 Standard Implementation from scratch."""
    
#     # Fractional parts of cube roots of first 64 primes
#     K = [
#         0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
#         0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
#         0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
#         0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
#         0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
#         0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
#         0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
#         0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
#     ]

#     @staticmethod
#     def _rotr(x, n):
#         return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

#     @staticmethod
#     def hash(message: str) -> str:
#         # Message Padding
#         msg = bytearray(message, 'utf-8')
#         length = len(msg) * 8
#         msg.append(0x80)
#         while (len(msg) * 8) % 512 != 448:
#             msg.append(0x00)
#         msg += length.to_bytes(8, 'big')

#         # Initial Hash Values (Square roots of first 8 primes)
#         h = [
#             0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
#             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
#         ]

#         # Compression Loop
#         for i in range(0, len(msg), 64):
#             w = [0] * 64
#             chunk = msg[i:i+64]
#             for j in range(16):
#                 w[j] = int.from_bytes(chunk[j*4:j*4+4], 'big')

#             for j in range(16, 64):
#                 s0 = SHA256._rotr(w[j-15], 7) ^ SHA256._rotr(w[j-15], 18) ^ (w[j-15] >> 3)
#                 s1 = SHA256._rotr(w[j-2], 17) ^ SHA256._rotr(w[j-2], 19) ^ (w[j-2] >> 10)
#                 w[j] = (w[j-16] + s0 + w[j-7] + s1) & 0xFFFFFFFF

#             a, b, c, d, e, f, g, h_v = h

#             for j in range(64):
#                 S1 = SHA256._rotr(e, 6) ^ SHA256._rotr(e, 11) ^ SHA256._rotr(e, 25)
#                 ch = (e & f) ^ ((~e) & g)
#                 t1 = (h_v + S1 + ch + SHA256.K[j] + w[j]) & 0xFFFFFFFF
#                 S0 = SHA256._rotr(a, 2) ^ SHA256._rotr(a, 13) ^ SHA256._rotr(a, 22)
#                 maj = (a & b) ^ (a & c) ^ (b & c)
#                 t2 = (S0 + maj) & 0xFFFFFFFF

#                 h_v, g, f, e, d, c, b, a = g, f, e, (d + t1) & 0xFFFFFFFF, c, b, a, (t1 + t2) & 0xFFFFFFFF

#             h = [(x + y) & 0xFFFFFFFF for x, y in zip(h, [a, b, c, d, e, f, g, h_v])]

#         return "".join(f"{x:08x}" for x in h)

# class CryptoMath:
#     @staticmethod
#     def modular_inverse(a: int, m: int) -> int:
#         m0, y, x = m, 0, 1
#         while a > 1:
#             q = a // m
#             a, m = m, a % m
#             y, x = x - q * y, y
#         return x + m0 if x < 0 else x

#     @staticmethod
#     def merkle_root(leaves: List[str]) -> str:
#         if not leaves: return ""
#         # Using our own scratch-built SHA256
#         hashes = [SHA256.hash(str(leaf)) for leaf in leaves]
#         while len(hashes) > 1:
#             if len(hashes) % 2 != 0: hashes.append(hashes[-1])
#             hashes = [SHA256.hash(hashes[i] + hashes[i+1]) for i in range(0, len(hashes), 2)]
#         return hashes[0]

# class EthereumEngine:
#     @staticmethod
#     def gas_to_usd(gas_used: int, gwei: float, eth_price: float) -> float:
#         return (gas_used * gwei * 1e-9) * eth_price

#     @staticmethod
#     def wei_to_eth(wei: int) -> float:
#         return wei / 1e18

# class ECC:
#     def __init__(self, a: int, b: int, p: int):
#         self.a, self.b, self.p = a, b, p

#     def is_on_curve(self, x: int, y: int) -> bool:
#         return (y**2 - (x**3 + self.a * x + self.b)) % self.p == 0
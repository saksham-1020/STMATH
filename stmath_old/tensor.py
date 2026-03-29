# import math

# class Tensor:
#     def __init__(self, data, requires_grad=False):
#         self.data = data
#         self.requires_grad = requires_grad
#         self.grad = 0.0 if requires_grad else None
#         self._backward = lambda: None
#         self._prev = set()

#     @property
#     def shape(self):
#         def get_shape(lst):
#             if not isinstance(lst, list): return []
#             return [len(lst)] + get_shape(lst[0])
#         return get_shape(self.data)

#     def __add__(self, other):
#         other = other if isinstance(other, Tensor) else Tensor(other)
#         out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
#         out._prev = {self, other}
#         def _backward():
#             if self.requires_grad: self.grad += out.grad
#             if other.requires_grad: other.grad += out.grad
#         out._backward = _backward
#         return out

#     def __mul__(self, other):
#         other = other if isinstance(other, Tensor) else Tensor(other)
#         out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
#         out._prev = {self, other}
#         def _backward():
#             if self.requires_grad: self.grad += other.data * out.grad
#             if other.requires_grad: other.grad += self.data * out.grad
#         out._backward = _backward
#         return out

#     def backward(self):
#         topo = []
#         visited = set()
#         def build_topo(v):
#             if v not in visited:
#                 visited.add(v)
#                 for child in v._prev:
#                     build_topo(child)
#                 topo.append(v)
#         build_topo(self)
#         self.grad = 1.0
#         for node in reversed(topo):
#             node._backward()

#     def __repr__(self):
#         return f"ST-Tensor({self.data}, shape={self.shape})"
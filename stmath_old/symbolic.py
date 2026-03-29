# import math

# class Symbol:
#     def __init__(self, name):
#         self.name = name

#     def __add__(self, other):
#         return Expression('+', self, other)

#     def __mul__(self, other):
#         return Expression('*', self, other)

#     def __pow__(self, other):
#         return Expression('**', self, other)

#     def __repr__(self):
#         return self.name

# class Expression:
#     def __init__(self, op, left, right):
#         self.op = op
#         self.left = left
#         self.right = right

#     def diff(self, var):
#         if self.op == '+':
#             return Expression('+', self.left.diff(var) if hasattr(self.left, 'diff') else Constant(0), 
#                              self.right.diff(var) if hasattr(self.right, 'diff') else Constant(0))
#         elif self.op == '*':
#             # Product Rule: (f*g)' = f'g + fg'
#             term1 = Expression('*', self.left.diff(var), self.right)
#             term2 = Expression('*', self.left, self.right.diff(var))
#             return Expression('+', term1, term2)
#         elif self.op == '**':
#             # Power Rule: (x^n)' = n*x^(n-1)
#             return Expression('*', Constant(self.right), Expression('**', self.left, Constant(self.right - 1)))

#     def __repr__(self):
#         return f"({self.left} {self.op} {self.right})"

# class Constant:
#     def __init__(self, value):
#         self.value = value
#     def __repr__(self):
#         return str(self.value)
#     def diff(self, var):
#         return Constant(0)
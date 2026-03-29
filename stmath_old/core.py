# import math
# from .engine import Value

# def add(a, b):
#     a = a if isinstance(a, Value) else Value(a)
#     b = b if isinstance(b, Value) else Value(b)
#     return a + b

# def sub(a, b):
#     a = a if isinstance(a, Value) else Value(a)
#     b = b if isinstance(b, Value) else Value(b)
#     return a + (-b)

# def mul(a, b):
#     a = a if isinstance(a, Value) else Value(a)
#     b = b if isinstance(b, Value) else Value(b)
#     return a * b

# def div(a, b):
#     if b == 0: raise ZeroDivisionError("Division by zero")
#     a = a if isinstance(a, Value) else Value(a)
#     return a * (b**-1)

# def square(x):
#     x = x if isinstance(x, Value) else Value(x)
#     return x**2

# def cube(x):
#     x = x if isinstance(x, Value) else Value(x)
#     return x**3

# def sqrt(x):
#     if x < 0: raise ValueError("Negative sqrt")
#     x = x if isinstance(x, Value) else Value(x)
#     return x**0.5

# def power(x, n):
#     x = x if isinstance(x, Value) else Value(x)
#     return x**n

# def percent(part, whole):
#     if whole == 0: raise ZeroDivisionError("Zero denominator")
#     return (part / whole) * 100.0

# def percent_change(old, new):
#     if old == 0: raise ZeroDivisionError("Zero base")
#     return ((new - old) / old) * 100.0

# from .engine import Value

# def ln(x):
#     x = Value(x) if not isinstance(x, Value) else x
#     return x.log_custom()

# def log10(x):
#     # ln(x) / ln(10) logic without math.log10
#     ln_x = ln(x)
#     ln_10 = ln(10).data
#     return ln_x.data / ln_10
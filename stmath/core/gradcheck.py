from .value import Value

def grad_check(f, x, eps=1e-5):

    # numerical gradient
    x1 = x + eps
    x2 = x - eps
    num_grad = (f(x1) - f(x2)) / (2 * eps)

    # autograd gradient
    x_val = Value(x)
    y = f(x_val)
    y.backward()
    auto_grad = x_val.grad

    return {
        "numerical": num_grad,
        "autograd": auto_grad,
        "difference": abs(num_grad - auto_grad)
    }
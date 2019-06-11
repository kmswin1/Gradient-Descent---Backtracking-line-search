import numpy as np

def min_gd(fun, x0, grad, args=()):
    # write your code here
    alpha = 0.3
    beta = 0.8
    dx = -grad(x0,*args)

    while (np.max(np.abs(dx)) > 1e-07):
        t = 1
        while fun(x0+t*dx,*args) > fun(x0,*args) - alpha*t*dx.T@dx:
            t *= beta
        x0 = x0 + t*dx
        dx = -grad(x0,*args)

    return x0
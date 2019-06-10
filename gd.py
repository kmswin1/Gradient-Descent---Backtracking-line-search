import numpy as np

def min_gd(fun, x0, grad, args=()):
    # write your code here
    alpha = 0.3
    beta = 0.8
    t = 1
    dx = x0*0.0001
    cur_gd = fun(x0+dx,*args)
    opt_gd = cur_gd + 1
    while opt_gd > cur_gd:
        if opt_gd - cur_gd <= 10e-7:
            break
        opt_gd = cur_gd
        cur_gd = fun(x0,*args) + alpha*t*grad(x0,*args).T@dx
        t *= beta

    return x0+t*dx
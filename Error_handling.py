import numpy as np

def error_propagation(x, grad_g, C):
    """
    function that calculates the variance of a given quantity as a function of others

    :param x: variable of the given function g - that describes the quantity in function of the others
    :param grad_g: function that returns the gradient of the function g
    :type grad_g: function
    :param C: 1D or 2D array (case in which C is the covariance matrix) of the x variable variance
    :type C: numpy array
    """

    if C.ndim == 1:
        Cl = np.identity(len(C))*C
    elif C.ndim == 2:
        Cl = C.copy( )
    
    return (grad_g(x) @ Cl @ grad_g(x).reshape(-1, 1))[0]


def grad_f(f, x, params = None, h = (np.finfo(float).eps)**(1/3)):
    """
    f numerical gradient (by finite differences method). f must operates point-wise in x (for a new dimension of it).
    x must be at least 1d numpy array, which elements are the multivariables. The others are only additional dimensions of data.

    The idea of this gradient function is to extend x in a new dimension, in which the difference step will be applied for each element at a time (partially).
    So, the finite difference is applied once (vectorially) in x.

    :param f: function (callable).
    :type f: function.
    :param x: vector (array) with variable's values.
    :type x: numpy.ndarray.
    :param h: difference step.
    :type h: float.
    """

    x = x[np.newaxis] # extending x dimension
    h = x*h + (x == 0)*h # avoid zero step

    X = np.swapaxes(x, 0, 1) * np.ones(x.shape)
    n = x.shape[1]
    I = np.eye(n)
    I = I[(...,) + (None,)*(x.ndim - 2)]
    print(I.shape)
    print((n, n) + x.shape[2:])
    I = np.broadcast_to(I, (n, n) + x.shape[2:])
    H = I * h

    if params is None:
        return (f(X + H) - f(X - H))/(2*h)[0]
    else:
        return (f(X + H, params) - f(X - H, params))/(2*h)[0]


def Var(f, x, sigma_x, params = None):
    """
    Variance of the function f with respect to the quantity q.
    
    :param x: variable
    :type x: numpy.ndarray
    :param sigma_x: variable uncertainty
    :type sigma_x: numpy.ndarray
    """

    sigma_x = sigma_x[np.newaxis]
    C = np.swapaxes(sigma_x, 0, 1) * sigma_x

    if params is not None:
        g_x = grad_f(f, x, params)[np.newaxis]
    else:
        g_x = grad_f(f, x)[np.newaxis]

    return np.einsum("in...,nk...,kj...->ij...", g_x, C, np.swapaxes(g_x, 0, 1))[0, 0]


def dev(f, x, sigma_x, params = None):
    """
    Standard deviation of the function f with respect to the quantity q.
    
    :param x: variable
    :type x: numpy.ndarray
    :param sigma_x: variable uncertainty
    :type sigma_x: numpy.ndarray
    """

    if params is not None:
        return np.sqrt(Var(f, x, sigma_x, params))
    else:
        return np.sqrt(Var(f, x, sigma_x))


def evaluate(f, x, sigma_x, params = None):
    """
    Returns the evaluation of q by function f and the propagated deviation.

    :param f: function (callable).
    :type f: function.
    :param x: variable
    :type x: numpy.ndarray
    :param sigma_x: variable uncertainty
    :type sigma_x: numpy.ndarray
    """

    if params is not None:
        return (f(x, params), dev(f, x, sigma_x, params))
    else:
        return (f(x), dev(f, x, sigma_x))

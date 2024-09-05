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

def num_grad_g(f, h = np.sqrt(np.finfo(float).eps)):
    """
    g numerical gradient

    :param f: the function to calculate the gradient.
    :type f: function.
    :param h: the step to calculate the numerical gradient.
    :type h: float.
    :return: the numerical gradient of g function.
    :rtype: function.
    """

    def grad_g(x):

        return (f(x + h) - f(x))/h

    return grad_g


import numpy as np


class Quantity:
    """
    Quantity class to represent a value with its associated uncertainty.
    """

    def __init__(self, x = None, sigma_x = None, name = None):
        """
        Initialize a Quantity instance.

        :param x: value of the quantity.
        :type x: float or list or numpy.ndarray.
        :param sigma_x: standard deviation of the quantity.
        :type sigma_x: float or list or numpy.ndarray.
        :param name: name of the quantity.
        :type name: string.
        """

        self.x = np.atleast_1d(x)
        self.sigma_x = np.atleast_1d(sigma_x)
        self.name = name

    def __str__(self):

        return self.name + '\n' + '\n'.join([f'{xi} ± {si}' for xi, si in zip(np.ravel(self.x), np.ravel(self.sigma_x))])

    def __repr__(self):
        
        return '\n'.join([f'{xi} ± {si}' for xi, si in zip(np.ravel(self.x), np.ravel(self.sigma_x))])

    def __len__(self):

        return len(self.x)
    
    def __getitem__(self, i):

        return Quantity(self.x[i], self.sigma_x[i])

    def __setitem__(self, i, q):
        self.x[i] = q.x
        self.sigma_x[i] = q.sigma_x

    def __add__(self, q):
        if isinstance(q, Quantity):
            return Quantity(self.x + q.x, np.sqrt(self.sigma_x**2 + q.sigma_x**2))
        else:
            return Quantity(self.x + q, self.sigma_x)
        
    def __radd__(self, q):
        if isinstance(q, Quantity):
            return Quantity(self.x + q.x, np.sqrt(self.sigma_x**2 + q.sigma_x**2))
        else:
            return Quantity(self.x + q, self.sigma_x)
    
    def __iadd__(self, q):
        if isinstance(q, Quantity):
            self.x += q.x
            self.sigma_x = np.sqrt(self.sigma_x**2 + q.sigma_x**2)
        else:
            self.x += q
        
        return self
    
    def __pos__(self):

        return self
    
    def __neg__(self):

        return Quantity(-self.x, self.sigma_x)

    def __sub__(self, q):
        if isinstance(q, Quantity):
            return Quantity(self.x - q.x, np.sqrt(self.sigma_x**2 + q.sigma_x**2))
        else:
            return Quantity(self.x - q, self.sigma_x)

    def __rsub__(self, q):
        if isinstance(q, Quantity):
            return Quantity(q.x - self.x, np.sqrt(self.sigma_x**2 + q.sigma_x**2))
        else:
            return Quantity(q - self.x, self.sigma_x)

    def __isub__(self, q):
        if isinstance(q, Quantity):
            self.x -= q.x
            self.sigma_x = np.sqrt(self.sigma_x**2 + q.sigma_x**2)
        else:
            self.x -= q
        
        return self

    def __mul__(self, q):
        if isinstance(q, Quantity):
            return Quantity(self.x * q.x, np.sqrt((q.x*self.sigma_x)**2 + (self.x*q.sigma_x)**2))
        else:
            return Quantity(self.x * q, self.sigma_x * np.abs(q))

    def __rmul__(self, q):
        if isinstance(q, Quantity):
            return Quantity(q.x * self.x, np.sqrt((self.x*q.sigma_x)**2 + (q.x*self.sigma_x)**2))
        else:
            return Quantity(q * self.x, self.sigma_x * np.abs(q))

    def __imul__(self, q):
        if isinstance(q, Quantity):
            self.x *= q.x
            self.sigma_x = np.sqrt((q.x*self.sigma_x)**2 + (self.x*q.sigma_x)**2)
        else:
            self.x *= q
            self.sigma_x *= np.abs(q)

        return self

    def __truediv__(self, q):
        if isinstance(q, Quantity):
            return Quantity(self.x / q.x, np.sqrt((self.sigma_x/q.x)**2 + (q.sigma_x*self.x/q.x**2)**2))
        else:
            return Quantity(self.x / q, self.sigma_x / np.abs(q))
    
    def __rtruediv__(self, q):
        if isinstance(q, Quantity):
            return Quantity(q.x/self.x, np.sqrt((q.sigma_x/self.x)**2 + (self.sigma_x*q.x/self.x**2)**2))
        else:
            return Quantity(q/self.x, np.abs(q)*self.sigma_x/self.x**2)

    def __itruediv__(self, q):
        if isinstance(q, Quantity):
            self.x /= q.x
            self.sigma_x = np.sqrt((self.sigma_x/q.x)**2 + (q.sigma_x*self.x/q.x**2)**2)
        else:
            self.x /= q
            self.sigma_x /= np.abs(q)

        return self

    def __pow__(self, q):

        return Quantity(self.x**q, self.sigma_x*q*self.x**(q - 1))

    def __lt__(self, q):
        if isinstance(q, Quantity):
            return self.x < q.x
        else:
            return self.x < q
    
    def __le__(self, q):
        if isinstance(q, Quantity):
            return self.x <= q.x
        else:
            return self.x <= q

    def __gt__(self, q):
        if isinstance(q, Quantity):
            return self.x > q.x
        else:
            return self.x > q

    def __ge__(self, q):
        if isinstance(q, Quantity):
            return self.x >= q.x
        else:
            return self.x >= q

    def __eq__(self, q):
        if isinstance(q, Quantity):
            return self.x == q.x
        else:
            return self.x == q

    def __ne__(self, q):
        if isinstance(q, Quantity):
            return self.x != q.x
        else:
            return self.x != q

    def full(self, n):
        
        return Quantity(np.full(n, self.x), np.full(n, self.sigma_x))
    
    def full_like(self, q):

        return Quantity(np.full_like(q.x, self.x), np.full_like(q.sigma_x, self.sigma_x))
    
    def zeros_like(self):

        return Quantity(np.zeros_like(self.x), np.zeros_like(self.sigma_x))
    
    def ones_like(self):

        return Quantity(np.ones_like(self.x), np.zeros_like(self.sigma_x))


def quantity_array(v):

    n = len(v)
    l = [None]*n
    sigma_l = [None]*n

    for i in range(n):
        l[i] = v[i].x
        sigma_l[i] = v[i].sigma_x

    return Quantity(l, sigma_l)


def sin(q):

    return Quantity(np.sin(q.x), np.abs(np.cos(q.x))*q.sigma_x)

def cos(q):

    return Quantity(np.cos(q.x), np.abs(np.sin(q.x))*q.sigma_x)

def tan(q):

    return Quantity(np.tan(q.x), 1/np.cos(q.x)**2*q.sigma_x)

def atan(q):

    return Quantity(np.atan(q.x), 1/(1 + q.x**2)*q.sigma_x)

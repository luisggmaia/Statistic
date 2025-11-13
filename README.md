# Statistic

This is a personal collection of functions and classes that are very useful in data handling.

### Its use

This set of tools provides essential functions and classes for operating within Error Theory.

### `quantity.py`

This module contains the `Quantity` class, which represents a quantity with its value and std. deviation. I made it math-compatible through to Python's magic methods, allowing uncertainties to be properly propagated as the calcualtions are performed. Just because of this, it becomes a powerful tool for data handling - no more worrying about uncertainty propagation.

### `Error_handling.py`

It includes a numerical gradient function and another function for the propagation of errors of a quantity described by a function. The num. gradient function works for n-dimensional arrays for which (at most) the first dimension represents multiple variables.
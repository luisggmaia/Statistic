class Quantity:

    def __init__(self, name = None, value = None, Var = None, sigma = None):
        """
        class that defines a quantity

        :param name: quantity name
        :type name: string
        :param value: quantity value
        :type value: float
        :param Var: quantity variance
        :type Var: float
        """
        self.name = name
        self.value = value
        self.Var = Var if sigma is None else sigma**2
        self.sigma = sigma if Var is None else Var**0.5

    def __str__(self):
        """
        string representation of the quantity
        """

        return f'{self.name} = {self.value} +- {self.sigma}'


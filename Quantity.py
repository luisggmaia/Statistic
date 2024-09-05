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
        self.Var = Var
        self.sigma = sigma

    def __str__(self):
        """
        string representation of the quantity
        """

        return f'{self.name} = {self.value} +- {self.sigma}'


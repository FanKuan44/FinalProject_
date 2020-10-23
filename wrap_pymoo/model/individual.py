import copy


class MyIndividual:
    def __init__(self, **kwargs) -> None:
        self.X = None
        self.hashX = None
        self.F = None
        self.CV = None
        self.G = None
        self.feasible = None
        self.data = kwargs

    def set(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        elif key in self.data:
            self.data[key] = value

    def copy(self):
        ind = copy.copy(self)
        ind.data = self.data.copy()
        return ind

    def get(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self.data:
            return self.data[key]
        else:
            return None

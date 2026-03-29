class Tensor:

    def __init__(self, data):
        self.data = data

    def shape(self):
        if isinstance(self.data, list):
            if isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)
        return (1,)

    def __add__(self, other):
        if isinstance(self.data, list):
            return Tensor([a + b for a, b in zip(self.data, other.data)])
        return Tensor(self.data + other.data)

    def __mul__(self, other):
        if isinstance(self.data, list):
            return Tensor([a * b for a, b in zip(self.data, other.data)])
        return Tensor(self.data * other.data)

    def sum(self):
        if isinstance(self.data, list):
            return sum(self.data)
        return self.data

    def mean(self):
        if isinstance(self.data, list):
            return sum(self.data) / len(self.data)
        return self.data

    def relu(self):
        if isinstance(self.data, list):
            return Tensor([max(0, x) for x in self.data])
        return Tensor(max(0, self.data))

    def __repr__(self):
        return f"Tensor({self.data})"
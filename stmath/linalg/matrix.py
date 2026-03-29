class Matrix:

    def __init__(self, data):
        self.data = data

    def shape(self):
        return (len(self.data), len(self.data[0]))

    def __repr__(self):
        return f"Matrix({self.data})"
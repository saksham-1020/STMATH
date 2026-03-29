class Validator:

    @staticmethod
    def check(X, y):
        if len(X) == 0:
            raise ValueError("Empty dataset")

        if len(X) != len(y):
            raise ValueError("X and y size mismatch")

        for row in X:
            for v in row:
                if v is None or v != v:
                    raise ValueError("Invalid value detected (NaN/None)")

        for v in y:
            if v is None or v != v:
                raise ValueError("Invalid target detected")
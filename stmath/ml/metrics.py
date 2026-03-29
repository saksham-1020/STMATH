class Metrics:

    # =============================
    # REGRESSION METRICS
    # =============================

    @staticmethod
    def mse(y_true, y_pred):
        return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

    @staticmethod
    def rmse(y_true, y_pred):
        return Metrics.mse(y_true, y_pred) ** 0.5

    @staticmethod
    def mae(y_true, y_pred):
        return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


    # =============================
    # CLASSIFICATION METRICS
    # =============================

    @staticmethod
    def accuracy(y_true, y_pred):
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return correct / len(y_true)


    @staticmethod
    def precision(y_true, y_pred):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        return tp / (tp + fp) if (tp + fp) != 0 else 0


    @staticmethod
    def recall(y_true, y_pred):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        return tp / (tp + fn) if (tp + fn) != 0 else 0


    @staticmethod
    def f1_score(y_true, y_pred):
        p = Metrics.precision(y_true, y_pred)
        r = Metrics.recall(y_true, y_pred)
        return (2 * p * r) / (p + r) if (p + r) != 0 else 0
    
    @staticmethod
    def r2(y_true, y_pred):
        mean_y = sum(y_true) / len(y_true)

        ss_tot = sum((yi - mean_y) ** 2 for yi in y_true)
        ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))

        return 1 - (ss_res / ss_tot if ss_tot != 0 else 0)
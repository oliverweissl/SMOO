import numpy as np
from pymoo.core.repair import Repair


class BudgetRepair(Repair):
    def __init__(self, budget_max=1.0, mode="multi", n_img=0, n_txt=0):
        super().__init__()
        self.budget_max = budget_max
        self.mode = mode
        self.n_img = n_img
        self.n_txt = n_txt

    def _do(self, problem, X, **kwargs):
        """Clip values to [0, 1] and rescale any individual that exceeds the budget.

        :param problem: pymoo problem instance (unused).
        :param X: Population matrix ``(n_samples, n_var)`` modified in place.
        :param kwargs: Additional keyword arguments (unused).
        :returns: Repaired population matrix.
        """
        np.clip(X, 0.0, 1.0, out=X)
        if self.mode == "multi":
            for block in (X[:, : self.n_img], X[:, self.n_img :]):
                sums = block.sum(axis=1, keepdims=True)
                over = (sums > self.budget_max).flatten()
                if over.any():
                    block[over] = block[over] / sums[over] * self.budget_max
        else:
            sums = X.sum(axis=1, keepdims=True)
            over = (sums > self.budget_max).flatten()
            if over.any():
                X[over] = X[over] / sums[over] * self.budget_max
        return X

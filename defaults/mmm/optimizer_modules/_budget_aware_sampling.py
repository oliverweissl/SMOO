import numpy as np
from pymoo.core.sampling import Sampling


class BudgetAwareSampling(Sampling):
    def __init__(self, budget_max=1.0, mode="multi", n_img=0, n_txt=0):
        super().__init__()
        self.budget_max = budget_max
        self.mode = mode
        self.n_img = n_img
        self.n_txt = n_txt

        self.n_var = {"multi": self.n_img + self.n_txt, "image": self.n_img, "text": self.n_txt}[
            self.mode
        ]

    def _do(self, problem, n_samples, **kwargs):
        """Sample an initial population where each individual respects the budget constraint.

        :param problem: pymoo problem instance (unused).
        :param n_samples: Number of individuals to sample.
        :param kwargs: Additional keyword arguments (unused).
        :returns: Population matrix of shape ``(n_samples, n_var)`` with budget-constrained values.
        """

        X = np.zeros((n_samples, self.n_var))
        n = self.n_var
        for i in range(n_samples):
            X[i] = np.random.dirichlet(np.ones(n)) * np.random.uniform(0, self.budget_max)
        return X

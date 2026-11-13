
import numpy as np

class SigmaPriorRegressor:

    def __init__(self, base, sigma_col="SUB_hat_log_sigma"):
        self.base, self.sigma_col = base, sigma_col

    def _sigma(self, X):
        s = X[self.sigma_col].astype(float).values
        return np.where(np.isnan(s), self.fill_, s)

    def fit(self, X, y):
        s = X[self.sigma_col].astype(float).values
        ok = ~np.isnan(s)
        self.fill_ = float(np.nanmean(s)) if ok.any() else 0.0
        self.a_, self.b_ = np.polyfit(s[ok], np.asarray(y)[ok], 1) if ok.sum() > 2 else (0.0, float(np.mean(y)))
        self.base.fit(X, np.asarray(y) - (self.a_ * self._sigma(X) + self.b_))
        return self

    def predict(self, X):
        return self.base.predict(X) + self.a_ * self._sigma(X) + self.b_

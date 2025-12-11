# Test: CausalForestDML med econml
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# Simuler data
np.random.seed(42)
n = 500
X = np.random.normal(0, 1, (n, 3))
W = np.random.normal(0, 1, (n, 2))
T = np.random.normal(0, 1, n)  # Continuous treatment
Y = 2 * T + X[:, 0] + W[:, 0] + np.random.normal(0, 1, n)

# Model
est = CausalForestDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestRegressor(),
    cv=3,
    random_state=42
)
est.fit(Y, T, X=X, W=W)
tau_hat = est.effect(X=X)

print("CausalForestDML virker! Eksempel på tau_hat:", tau_hat[:5])
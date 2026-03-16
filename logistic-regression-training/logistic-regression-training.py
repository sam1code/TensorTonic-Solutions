import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=10000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y).reshape(-1)  # Ensure shape (n_samples,)
    
    num_samples, num_features = X.shape
    w = np.zeros(num_features)  # Start from zeros for reproducibility
    b = 0.0

    for step in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        
        dw = (X.T @ (p - y)) / num_samples
        db = np.sum(p - y) / num_samples
        
        w -= lr * dw
        b -= lr * db

    return w, b
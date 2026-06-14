import numpy as np
from scipy.optimize import minimize

def flatten_X(X):
    return X.ravel()

def unflatten_X(x_flat, shape):
    return x_flat.reshape(shape)

def phi_squared(X, R, W=None):
    """
    Compute objective function Φ.

    Parameters
    ----------
    X : ndarray (n, d)
        Embedding coordinates
    R : ndarray (n, n)
        Correlation matrix
    W : ndarray (n, n), optional
        Weights matrix

    Returns
    -------
    float
        Objective value
    """
    diff = R - X @ X.T

    if W is None:
        return 0.5 * np.sum(diff**2)

    return 0.5 * np.sum(W * diff**2)

def gradient_phi(X, R, W=None):
    """
    Gradient of Φ with respect to X.

    Parameters
    ----------
    X : ndarray (n, d)
    R : ndarray (n, n)
    W : ndarray (n, n), optional

    Returns
    -------
    ndarray (n, d)
        Gradient
    """
    XXT = X @ X.T
    if W is None:
        grad = -2 * (R - XXT) @ X
    else:
        grad = -2 * (W * (R - XXT)) @ X
    return grad

def phi_objective(x_flat, R, d, W=None):
    n = R.shape[0]
    X = x_flat.reshape((n, d))
    return phi_squared(X, R, W)

def gradient_objective(x_flat, R, d, W=None):
    n = R.shape[0]
    X = x_flat.reshape((n, d))
    grad = gradient_phi(X, R, W)
    return grad.ravel()

def optimize_X(
    R,
    d,
    W=None,
    use_spectral_init=True
):
    """
    Optimize X using L-BFGS-B
    """

    n = R.shape[0]

    if use_spectral_init:
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        X0 = eigvecs[:, :d] @ np.diag(
            np.sqrt(np.maximum(eigvals[:d], 0))
        )
    else:
        X0 = np.random.normal(size=(n, d))

    x0 = flatten_X(X0)

    result = minimize(
        phi_objective,
        x0,
        args=(R, d, W),
        jac=gradient_objective,
        method="L-BFGS-B",
    )

    X_opt = unflatten_X(result.x, (n, d))

    return X_opt, result.fun

def scan_dimensions(R, d_max=10, W=None):
    """
    Evaluate Φ for multiple dimensions and detect elbow.

    Parameters
    ----------
    R : ndarray (n, n)
        Correlation matrix
    d_max : int
        Maximum dimension to test

    Returns
    -------
    X_values
        List of X final values for each dimension
    phi_values
        List of Φ values for each dimension
    """
    phi_values = []
    X_values = []

    for d in range(1, d_max + 1):
        print(f"Scanning dimension {d}...")
        X, phi_val = optimize_X(R, d, W=W)
        phi_values.append(phi_val)
        X_values.append(X)

    return X_values, phi_values
import numpy as np
from scipy.optimize import minimize, basinhopping

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
        return 2 * np.sum(diff**2)

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

def get_initial_X(R, d, use_spectral_init):

    n = R.shape[0]

    if use_spectral_init:
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        X0 = eigvecs[:, :d] @ np.diag(
            np.sqrt(np.maximum(eigvals[:d], 0))
        ) * 0.1
    else:
        X0 = np.random.normal(size=(n, d)) * 5

    return X0

def optimize_X_LBFGS(R, d, use_spectral_init, W):
    """
    Optimize X using L-BFGS-B
    """

    n = R.shape[0]

    X0 = get_initial_X(R, d, use_spectral_init)

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

def optimize_X_annealing(R, d, use_spectral_init, W, niter=50, temperature=1.0):
    """
    Optimize X using a Simulated Annealing framework via Basin-Hopping.

    Parameters
    ----------
    R : ndarray (n, n)
        Correlation matrix.
    d : int
        Target embedding dimension.
    W : ndarray (n, n), optional
        Weights matrix.
    use_spectral_init : bool, default=True
        If True, initializes via eigendecomposition. If False, uses random normal.
    niter : int, default=50
        The number of basin-hopping simulated annealing iterations (cycles).
    temperature : float, default=1.0
        The "temperature" parameter controlling the acceptance probability 
        for uphill moves.

    Returns
    -------
    X_opt : ndarray (n, d)
        Optimized embedding matrix coordinates.
    fun_val : float
        Final optimized Φ value.
    """
    n = R.shape[0]

    X0 = get_initial_X(R, d, use_spectral_init)

    x0 = flatten_X(X0)

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "args": (R, d, W),
        "jac": gradient_objective
    }

    result = basinhopping(
        phi_objective,
        x0,
        niter=niter,
        T=temperature,
        minimizer_kwargs=minimizer_kwargs,
        seed=42
    )

    X_opt = unflatten_X(result.x, (n, d))

    return X_opt, result.fun

def optimize_X_gradient_descent(R, d, use_spectral_init, W, learning_rate=0.001, epochs=2000, tol=1e-6):
    """
    Optimize X using vanilla Gradient Descent.
    
    Parameters
    ----------
    R : ndarray (n, n)
    d : int
    W : ndarray (n, n), optional
    learning_rate : float
        The step size (alpha). If this is too high, the loss will explode (NaN).
    epochs : int
        Maximum number of iterations.
    tol : float
        Early stopping threshold based on coordinate changes.
    """

    X = get_initial_X(R, d, use_spectral_init)
    
    for epoch in range(epochs):

        grad = gradient_phi(X, R, W)

        X_new = X - learning_rate * grad

        change = np.max(np.abs(X_new - X))
        X = X_new

        if epoch % 500 == 0:
            current_loss = phi_squared(X, R, W)
            print(f"  [GD Epoch {epoch}] Loss Phi: {current_loss:.4f}")
            
        if change < tol:
            print(f"  Gradient descent converged early at epoch {epoch}.")
            break

    final_loss = phi_squared(X, R, W)
    return X, final_loss

def scan_dimensions(R, minimize_method, use_spectral_init, d_max, W=None):
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
        if minimize_method == "lbfgs":
            X, phi_val = optimize_X_LBFGS(R, d, use_spectral_init, W)
        elif minimize_method == "simulated_annealing":
            X, phi_val = optimize_X_annealing(R, d, use_spectral_init, W)
        elif minimize_method == "gradient_descent":
            X, phi_val = optimize_X_gradient_descent(R, d, use_spectral_init, W)
        else:
            print("Invalid minimalization method")
            raise SystemExit()
        phi_values.append(phi_val)
        X_values.append(X)

    return X_values, phi_values
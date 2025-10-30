import numpy as np


def fit_polynomial(
    x: np.ndarray, y: np.ndarray, degree: int = 2
) -> tuple[np.ndarray, float]:
    """
    Fit a polynomial of specified degree using least squares.

    Args:
        x: Input x values
        y: Input y values
        degree: Polynomial degree (1=linear, 2=quadratic, 3=cubic)

    Returns:
        tuple: (coefficients, mean_squared_error)
    """
    # Fit polynomial using numpy's polyfit (least squares)
    coeffs = np.polyfit(x, y, degree)

    # Calculate predictions and error
    y_pred = np.polyval(coeffs, x)
    mse = np.mean((y - y_pred) ** 2)

    print(f"\n📐 Polynomial Fit (degree={degree})")
    print(f"   Coefficients: {coeffs}")
    print(f"   Mean Squared Error: {mse:.4f}")

    return coeffs, mse


def evaluate_model(
    x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, degree: int
) -> dict:
    """
    Evaluate the fitted model and compute metrics.

    Args:
        x: Input x values
        y: True y values
        coeffs: Polynomial coefficients
        degree: Polynomial degree

    Returns:
        dict: Metrics including MSE, RMSE, R²
    """
    y_pred = np.polyval(coeffs, x)

    # Calculate metrics
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # R² (coefficient of determination)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    metrics = {
        "degree": degree,
        "mse": mse,
        "rmse": rmse,
        "r_squared": r_squared,
        "num_coefficients": len(coeffs),
    }

    print(f"\n📊 Model Evaluation (degree={degree})")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²: {r_squared:.4f}")

    return metrics

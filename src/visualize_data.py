import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_data(
    x: np.ndarray, 
    y: np.ndarray, 
    coeffs: np.ndarray, 
    degree: int,
    output_path: str = "data/output.png"
) -> None:
    """
    Visualize the data and polynomial fit, save to file.

    Args:
        x: Input x values
        y: Input y values
        coeffs: Polynomial coefficients
        degree: Polynomial degree
        output_path: Path to save the plot
    """
    # Set matplotlib to not display plots (save only)
    plt.ioff()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot original data points
    ax.scatter(x, y, alpha=0.6, s=50, label="Data points", color="blue")
    
    # Generate smooth curve for polynomial fit
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = np.polyval(coeffs, x_smooth)
    
    # Plot polynomial fit
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
            label=f"Polynomial fit (degree {degree})")
    
    # Formatting
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Polynomial Regression (Degree {degree})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save to file
    output_file = Path(__file__).parent.parent / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"📊 Saved visualization to: {output_file}")

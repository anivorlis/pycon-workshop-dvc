from pathlib import Path

import numpy as np


def generate_data(
    num_points: int = 100,
    noise_level: float = 0.5,
    seed: int = 42,
    output_path: str = "data/input.csv",
) -> None:
    """
    Generate synthetic data from a known polynomial with added noise.

    The true underlying function is: y = 0.5*x^2 - 2*x + 1

    Args:
        num_points: Number of data points to generate
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
        output_path: Path to save the CSV file
    """
    np.random.seed(seed)

    # Generate x values uniformly in [-3, 3]
    x = np.linspace(-3, 3, num_points)

    # True underlying function (quadratic)
    y_true = 0.5 * x**2 - 2 * x + 1

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, num_points)
    y_noisy = y_true + noise

    # Create output path relative to project root
    output_file = Path(__file__).parent.parent / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    with open(output_file, "w") as f:
        f.write("x,y\n")
        for xi, yi in zip(x, y_noisy):
            f.write(f"{xi:.6f},{yi:.6f}\n")

    print(f"✅ Generated {num_points} noisy data points")
    print(f"📁 Saved to: {output_file}")

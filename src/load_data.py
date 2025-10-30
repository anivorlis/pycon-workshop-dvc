from pathlib import Path

import numpy as np


def load_data(input_path: str = "data/input.csv") -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file into numpy arrays.

    Args:
        input_path: Path to the CSV file containing x,y data

    Returns:
        tuple: (x_values, y_values) as numpy arrays
    """
    # Create input path relative to project root
    input_file = Path(__file__).parent.parent / input_path

    if not input_file.exists():
        raise FileNotFoundError(f"Data file not found: {input_file}")

    # Load data from CSV
    data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]

    print(f"✅ Loaded {len(x)} data points from {input_file}")

    return x, y

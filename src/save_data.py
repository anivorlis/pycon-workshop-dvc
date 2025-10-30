from pathlib import Path


def save_metrics(metrics: dict, output_path: str = "data/output.csv") -> None:
    """
    Save model metrics to CSV file.

    Args:
        metrics: Dictionary containing metrics from evaluate_model
        output_path: Path to save the metrics CSV
    """
    # Create output path relative to project root
    output_file = Path(__file__).parent.parent / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write metrics to CSV
    with open(output_file, "w") as f:
        f.write("metric,value\n")
        f.write(f"degree,{metrics['degree']}\n")
        f.write(f"mse,{metrics['mse']:.6f}\n")
        f.write(f"rmse,{metrics['rmse']:.6f}\n")
        f.write(f"r_squared,{metrics['r_squared']:.6f}\n")
        f.write(f"num_coefficients,{metrics['num_coefficients']}\n")

    print(f"📁 Saved metrics to: {output_file}")

from src.generate_data import generate_data
from src.load_data import load_data
from src.process_data import evaluate_model, fit_polynomial
from src.save_data import save_metrics
from src.visualize_data import visualize_data


def run_pipeline(degree: int = 2):
    """
    Run the complete polynomial fitting pipeline.

    This demonstrates a typical data science workflow:
    1. Generate data
    2. Load data
    3. Fit model
    4. Evaluate model
    5. Save metrics
    6. Visualize results

    Args:
        degree: Polynomial degree to use (default: 2)
    """
    print("🚀 Polynomial Fitting Pipeline")
    print("=" * 60)

    # Stage 1: Generate data
    print("\n📝 Stage 1: Generating Data")
    generate_data()

    # Stage 2: Load data
    print("\n📝 Stage 2: Loading Data")
    x, y = load_data()

    # Stage 3: Fit polynomial
    print(f"\n📝 Stage 3: Fitting Polynomial (degree={degree})")
    coeffs, mse = fit_polynomial(x, y, degree)

    # Stage 4: Evaluate model
    print("\n📝 Stage 4: Evaluating Model")
    metrics = evaluate_model(x, y, coeffs, degree)

    # Stage 5: Save metrics
    print("\n📝 Stage 5: Saving Metrics")
    save_metrics(metrics)

    # Stage 6: Visualize
    print("\n📝 Stage 6: Visualizing Results")
    visualize_data(x, y, coeffs, degree)

    print("\n✅ Pipeline complete!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    run_pipeline()

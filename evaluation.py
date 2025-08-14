"""
Evaluation module for house price prediction models.
Contains all metrics and evaluation logic with proper type handling.
"""

from typing import Any

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Percentage Error - robust to outliers."""
    return float(np.median(np.abs((y_true - y_pred) / y_true)) * 100)


def accuracy_within_pct(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 10) -> float:
    """Percentage of predictions within threshold% of actual price."""
    pct_errors = np.abs((y_true - y_pred) / y_true) * 100
    return float((pct_errors <= threshold).mean() * 100)


def log_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log-scale RMSE - handles heteroscedasticity."""
    return float(np.sqrt(np.mean((np.log(y_true) - np.log(y_pred)) ** 2)))


def measure_prediction_latency(pipeline, X_sample: np.ndarray, n_warmup: int = 3, n_measurements: int = 10) -> float:
    """Measure single prediction latency in milliseconds."""
    # Warmup predictions to ensure consistent timing
    for _ in range(n_warmup):
        pipeline.predict(X_sample[:1])
    
    # Measure actual prediction time
    times = []
    for _ in range(n_measurements):
        start = time.perf_counter()
        pipeline.predict(X_sample[:1])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    # Return median time (robust to outliers)
    return float(np.median(times))


def price_segment_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate variance in MAPE across price segments."""
    # Define price ranges based on market segments
    budget_mask = y_true < 400000  # Budget homes
    mid_mask = (y_true >= 400000) & (y_true < 800000)  # Mid-market
    luxury_mask = y_true >= 800000  # Luxury homes
    
    segment_mapes = []
    
    for mask in [budget_mask, mid_mask, luxury_mask]:
        if mask.sum() > 0:  # Only if we have samples in this range
            y_seg = y_true[mask]
            pred_seg = y_pred[mask]
            segment_mapes.append(mape(y_seg, pred_seg))
    
    # Return standard deviation of segment MAPEs
    return float(np.std(segment_mapes)) if len(segment_mapes) > 1 else 0.0


def calculate_comprehensive_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, split_name: str = "test"
) -> dict[str, Any]:
    """
    Calculate all metrics for a given prediction set with proper type conversion.

    Args:
        y_true: True values
        y_pred: Predicted values
        split_name: Name of the split (train/validation/test)

    Returns:
        Dictionary of all metrics with proper Python types for JSON serialization
    """
    # Base regression metrics
    base_metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        # Percentage-based metrics (more interpretable for house prices)
        "mape": mape(y_true, y_pred),
        "mdape": mdape(y_true, y_pred),
        # Accuracy thresholds (business-friendly metrics)
        "accuracy_within_10pct": accuracy_within_pct(y_true, y_pred, 10),
        "accuracy_within_15pct": accuracy_within_pct(y_true, y_pred, 15),
        "accuracy_within_20pct": accuracy_within_pct(y_true, y_pred, 20),
    }

    # Add price segment variance for test set
    if split_name == "test":
        base_metrics["price_segment_variance"] = price_segment_variance(y_true, y_pred)

    return base_metrics


def evaluate_predictions(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    pipeline=None,
    X_test=None,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate predictions across all splits with comprehensive metrics.

    Args:
        y_train, y_train_pred: Training true and predicted values
        y_val, y_val_pred: Validation true and predicted values
        y_test, y_test_pred: Test true and predicted values

    Returns:
        Dictionary with metrics for each split
    """
    results = {
        "train": calculate_comprehensive_metrics(y_train, y_train_pred, "train"),
        "validation": calculate_comprehensive_metrics(y_val, y_val_pred, "validation"),
        "test": calculate_comprehensive_metrics(y_test, y_test_pred, "test"),
    }
    
    # Add prediction latency if pipeline and X_test are provided
    if pipeline is not None and X_test is not None:
        latency = measure_prediction_latency(pipeline, X_test)
        results["test"]["prediction_latency_ms"] = latency
    
    return results


def get_summary_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Extract key metrics for registry summary with proper type conversion.

    Args:
        metrics: Full metrics dictionary from evaluate_predictions

    Returns:
        Summary metrics for registry storage
    """
    test_metrics = metrics["test"]

    summary = {
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_mape": test_metrics["mape"],
        "test_mdape": test_metrics["mdape"],
        "test_accuracy_15pct": test_metrics["accuracy_within_15pct"],
        "test_price_segment_variance": test_metrics["price_segment_variance"],
    }
    
    # Add latency if available
    if "prediction_latency_ms" in test_metrics:
        summary["test_prediction_latency_ms"] = test_metrics["prediction_latency_ms"]
    
    return summary


def format_metrics_for_display(metrics: dict[str, dict[str, Any]]) -> str:
    """
    Format metrics for CLI display with emojis and clear structure.

    Args:
        metrics: Full metrics dictionary

    Returns:
        Formatted string for display
    """
    test_metrics = metrics["test"]
    val_metrics = metrics["validation"]

    output = []

    # Primary metrics section
    output.append("🎯 Primary Metrics (Test Set):")
    output.append(f"  MAPE:                   {test_metrics['mape']:.1f}% (avg error)")
    output.append(f"  MdAPE:                  {test_metrics['mdape']:.1f}% (median error)")
    output.append(f"  MAE:                    ${test_metrics['mae']:,.0f} (avg dollar error)")
    output.append(f"  Within 15%:             {test_metrics['accuracy_within_15pct']:.1f}% of predictions")
    output.append(f"  Price Segment Variance: {test_metrics['price_segment_variance']:.1f}% (consistency across price ranges)")
    output.append(f"  R²:                     {test_metrics['r2']:.3f} (variance explained)")

    # Performance metrics
    if "prediction_latency_ms" in test_metrics:
        output.append("\n⚡ Performance:")
        output.append(f"  Single Prediction:      {test_metrics['prediction_latency_ms']:.1f}ms")
    
    # Validation comparison
    output.append("\n📊 Validation Comparison:")
    output.append(f"  Val MAPE:       {val_metrics['mape']:.1f}%")
    output.append(f"  Val MAE:        ${val_metrics['mae']:,.0f}")

    return "\n".join(output)


def compare_experiments_df(experiments_data: list) -> pd.DataFrame:
    """
    Create a comparison DataFrame from experiment metadata.

    Args:
        experiments_data: List of experiment dictionaries with metrics

    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []

    for exp in experiments_data:
        test_metrics = exp["metrics"]["test"]
        row = {
            "id": exp["id"],
            "name": exp["name"],
            "test_mape": test_metrics.get("mape", "N/A"),
            "test_mdape": test_metrics.get("mdape", "N/A"),
            "accuracy_15pct": test_metrics.get("accuracy_within_15pct", "N/A"),
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
            "price_segment_variance": test_metrics.get("price_segment_variance", "N/A"),
            "features": exp["features"]["count"],
            "data_hash": exp["features"]["data_hash"],
        }

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def determine_best_model(current_best: dict[str, Any], new_model: dict[str, Any]) -> dict[str, Any]:
    """
    Determine the best model between current best and new model.
    Uses MAPE if available, falls back to RMSE.

    Args:
        current_best: Current best model summary
        new_model: New model summary to compare

    Returns:
        The better model summary
    """
    if not current_best:
        return new_model

    # Use MAPE if both experiments have it (lower is better)
    if "test_mape" in new_model and "test_mape" in current_best:
        if new_model["test_mape"] < current_best["test_mape"]:
            return new_model
        else:
            return current_best

    # Fall back to MAE comparison (lower is better)
    new_mae = new_model.get("test_mae", float('inf'))
    current_mae = current_best.get("test_mae", float('inf'))
    
    if new_mae < current_mae:
        return new_model
    else:
        return current_best

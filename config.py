"""
Configuration settings for ML experiment system.
"""

# Data settings
DATA_CATALOG_DIR = "data_catalog"

DATA_SOURCES = {
    "sales_path": "data_catalog/sources/kc_house_data.csv",
    "demographics_path": "data_catalog/sources/zipcode_demographics.csv",
    "future_unseen_examples_path": "data_catalog/sources/future_unseen_examples.csv"
}

DATA_SPLIT = {
    "test_size": 0.2,
    "val_size": 0.25,
    "random_state": 42
}

# Registry settings
REGISTRY_EXPERIMENTS_DIR = "model_registry"
REGISTRY_FILE = "model_registry/registry.json"

# Quality gates for production promotion
QUALITY_GATES = {
    "max_mape": 15.0,              # Maximum MAPE %
    "min_accuracy_15pct": 65.0,    # Minimum % predictions within 15%
    "min_r2": 0.85,                # Minimum R-squared
    "max_latency_ms": 10.0,        # Maximum prediction latency
    "min_improvement_pct": 2.0     # Minimum % improvement over current production
}

# Model promotion settings
PROMOTION = {
    "auto_promote": True,          # Enable auto-promotion in API
    "require_all_gates": True,     # All gates must pass
    "allow_force_promote": True    # Allow manual override
}
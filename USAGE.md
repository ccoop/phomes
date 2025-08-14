# phData ML Engineer Project - Usage Guide

This guide walks through testing and exploring the machine learning housing price prediction system.

## Setup

```bash
docker build -t phdata-homes .

# Run interactive container
docker run -it phdata-homes bash
```
Inside the container, the `mla` CLI is available directly.

## Run API Tests

```bash
pytest tests/test_api.py -v
```

This comprehensive test suite will:
- Start the API server automatically
- Test input validation, functionality, and performance
- Run 12 tests including concurrent load testing
- Display performance metrics
- Clean up automatically

Expected result: **12/12 tests passed** with latency metrics.

## MLA CLI provides direct access to common ML workflows

### Data Management

#### Create Data Snapshot
```bash
mla data snapshot --note "Initial dataset for testing"
```
Creates a versioned snapshot with validation. Returns version ID (e.g., `v1`).

#### List Data Versions
```bash
mla data list
```
Shows all data versions with metadata, samples, features, and current version indicator.

### Model Training

#### List Available Models
```bash
mla list
```
Shows available model types:
- `knn_baseline` - K-Nearest Neighbors with robust scaling
- `random_forest` - Random Forest with standard scaling
- `gradient_boost` - Gradient Boosting with tuned parameters

#### Train Models
```bash
# Train with default parameters
mla train knn_baseline

# Train ALL models at once
mla train --all

# Train with custom parameters
mla train gradient_boost --params n_estimators=200 learning_rate=0.05

# Train with specific data version and notes
mla train random_forest --data v2 --note "Experiment with more trees"

# Train all models with custom parameters
mla train --all --params n_estimators=50 --note "Quick baseline experiments"
```

Each training run uses versioned data, generates unique experiment ID, runs comprehensive evaluation, and saves all artifacts.

#### Start API Server
```bash
# Start server on default port 8000
mla serve

# Start server on custom host/port
mla serve --host 127.0.0.1 --port 8080
```

### Model Registry & Management

#### List Models
```bash
mla models list
```
Shows experiment registry with:
- All experiments ranked by performance
- Current production model (🚀)
- Best performing model (⭐)
- Key metrics (MAPE, accuracy within 15%, MAE)

#### Show Model Details
```bash
# Show specific experiment details
mla models show gradient_boost_20250813_170200

# Show production model details
mla models show prod

# Show best model details
mla models show best
```

#### Compare Experiments
```bash
mla models compare gradient_boost_20250813_170200 random_forest_20250813_170233
```
Side-by-side comparison of model performance metrics.

#### Promote to Production
```bash
# Promote with metric gated validation
mla models promote gradient_boost_20250813_170200

```

Quality gates validate MAPE < 15%, accuracy within 15% > 65%, R² > 0.85, latency < 10ms, and minimum 2% improvement.

## Example Workflow

Complete demonstration workflow:

```bash
# 1. Create initial data snapshot
mla data snapshot --note "Clean dataset for baseline experiments"

# 2. Train all models at once
mla train --all --note "Initial baseline experiments"

# 3. View results and rankings
mla models list

# 4. Compare best models (use actual experiment IDs from registry)
mla models compare <experiment_id_1> <experiment_id_2>

# 5. Promote best model to production
mla models promote <best_experiment_id>

```

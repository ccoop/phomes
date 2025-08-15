#!/usr/bin/env python3
"""
Analyze feature correlations to identify redundant features
"""

import pandas as pd
from shared import catalog

print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test, version_id = catalog.load_version()

# Combine all data for correlation analysis
X_all = pd.concat([X_train, X_val, X_test])
print(f"Analyzing {X_all.shape[0]} samples with {X_all.shape[1]} features")

print("\nCalculating correlation matrix...")
corr_matrix = X_all.corr()


def find_high_correlations(corr_matrix, threshold=0.9):
    """Find feature pairs with correlation above threshold"""
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                high_corr_pairs.append(
                    {
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    }
                )

    return pd.DataFrame(high_corr_pairs).sort_values("correlation", ascending=False)


# Find correlations above different thresholds
print("\n=== HIGHLY CORRELATED FEATURES ===")
for threshold in [0.95, 0.90, 0.85]:
    high_corr = find_high_correlations(corr_matrix, threshold)
    print(f"\nFeatures with correlation >= {threshold}:")
    if len(high_corr) > 0:
        for _, row in high_corr.iterrows():
            print(f"  {row['feature1']:25s} <-> {row['feature2']:25s} : {row['correlation']:.3f}")
    else:
        print("  None found")

# Analyze feature groups
print("\n=== FEATURE GROUP ANALYSIS ===")

# Group features by type
house_features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "sqft_living15",
    "sqft_lot15",
]
location_features = ["lat", "long"]
demographic_features = [col for col in X_all.columns if col not in house_features + location_features]

print(f"\nHouse features ({len(house_features)}): {house_features}")
print(f"\nLocation features ({len(location_features)}): {location_features}")
print(f"\nDemographic features ({len(demographic_features)}): {', '.join(demographic_features[:5])}...")

# Analyze correlation with target
print("\n=== CORRELATION WITH TARGET (PRICE) ===")
y_all = pd.concat([y_train, y_val, y_test])
target_corr = X_all.corrwith(y_all).sort_values(ascending=False)

print("\nTop 15 features most correlated with price:")
for feature, corr in target_corr.head(15).items():
    print(f"  {feature:30s} : {corr:.3f}")

print("\nBottom 15 features least correlated with price:")
for feature, corr in target_corr.tail(15).items():
    print(f"  {feature:30s} : {corr:.3f}")

# Identify redundant features to remove
print("\n=== RECOMMENDED FEATURES TO EXCLUDE ===")

# Strategy: Remove one feature from highly correlated pairs
# Keep the one with higher correlation to target
features_to_exclude = set()

high_corr_095 = find_high_correlations(corr_matrix, 0.95)
for _, row in high_corr_095.iterrows():
    feat1, feat2 = row["feature1"], row["feature2"]
    # Keep the feature with higher correlation to target
    if abs(target_corr[feat1]) < abs(target_corr[feat2]):
        features_to_exclude.add(feat1)
    else:
        features_to_exclude.add(feat2)

# Also consider features with very low correlation to target
low_importance_features = target_corr[abs(target_corr) < 0.05].index.tolist()

print(f"\nFeatures to exclude due to high correlation with other features (>0.95):")
for feat in sorted(features_to_exclude):
    print(f"  - {feat}")

print(f"\nFeatures with very low correlation to price (<0.05):")
for feat in low_importance_features:
    print(f"  - {feat}: {target_corr[feat]:.3f}")

# Combine recommendations
all_exclude = sorted(list(features_to_exclude.union(set(low_importance_features))))
print(f"\n=== FINAL RECOMMENDATION ===")
print(f"Exclude {len(all_exclude)} features: {', '.join(all_exclude)}")
print(f"Keep {X_all.shape[1] - len(all_exclude)} features")

# Calculate expected feature importance from tree model
print("\n=== FEATURE IMPORTANCE (from existing gradient boost model) ===")
try:
    import pickle
    from pathlib import Path

    # Find a gradient boost model dynamically
    registry_dir = Path("model_registry")
    gb_models = list(registry_dir.glob("gradient_boost*/model.pkl"))

    if not gb_models:
        raise FileNotFoundError("No gradient boost models found")

    model_path = gb_models[0]  # Use the first one found
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Get feature importance from the gradient boosting model
    gb_model = pipeline.named_steps["model"]
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": gb_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop 15 most important features (by tree splits):")
    for _, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:30s} : {row['importance']:.4f}")

    print("\nBottom 15 least important features:")
    for _, row in feature_importance.tail(15).iterrows():
        print(f"  {row['feature']:30s} : {row['importance']:.4f}")

except Exception as e:
    print(f"Could not load gradient boost model: {e}")

print("\n=== SUGGESTED FEATURE SETS ===")
print("\n1. Minimal set (top correlated with price):")
minimal_features = target_corr.head(15).index.tolist()
print(f"   {', '.join(minimal_features)}")

print("\n2. Balanced set (remove highly correlated + low importance):")
balanced_features = [f for f in X_all.columns if f not in all_exclude]
print(f"   {len(balanced_features)} features (excluding: {', '.join(all_exclude)})")

print("\n3. Aggressive reduction (keep only top 20 by correlation):")
aggressive_features = target_corr.head(20).index.tolist()
print(f"   {', '.join(aggressive_features)}")

from shared import registry
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


@registry.experiment(name="knn_baseline", desc="Baseline KNN with robust scaling")
def knn_baseline(n_neighbors=5, weights="distance"):
    return Pipeline(
        [("scaler", RobustScaler()),
         ("model", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))]
    )


@registry.experiment(name="random_forest", desc="Random Forest with standard scaling")
def random_forest(n_estimators=100, max_depth=10, min_samples_split=5):
    return Pipeline(
        [("scaler", StandardScaler()),
         ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
            ),
        )]
    )


@registry.experiment(name="gradient_boost", desc="Gradient Boosting with tuned parameters")
def gradient_boost(n_estimators=150, learning_rate=0.1, max_depth=5):
    return Pipeline(
        [("scaler", RobustScaler()),
         ("model", GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            ),
        )]
    )

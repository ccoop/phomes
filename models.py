import config
from shared import registry
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


@registry.experiment(name="knn_baseline", desc="Baseline KNN with robust scaling")
def knn_baseline(n_neighbors=5, weights="uniform"):
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
            random_state=config.DATA_SPLIT["random_state"]
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
            random_state=config.DATA_SPLIT["random_state"],
            ),
        )]
    )


@registry.experiment(name="hist_gradient_boost", desc="Modern histogram-based gradient boosting")
def hist_gradient_boost(max_iter=200, max_depth=7, learning_rate=0.1):
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=config.DATA_SPLIT["random_state"]
    )


# Hyperparameter-tuned models
@registry.experiment(name="random_forest_tuned", desc="Random Forest with optimized hyperparameters")
def random_forest_tuned(n_estimators=200, max_depth=15, min_samples_split=4, min_samples_leaf=2):
    return Pipeline(
        [("scaler", StandardScaler()),
         ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=config.DATA_SPLIT["random_state"]
            ),
        )]
    )


@registry.experiment(name="gradient_boost_tuned", desc="Gradient Boosting with optimized hyperparameters")
def gradient_boost_tuned(n_estimators=200, learning_rate=0.08, max_depth=6, subsample=0.8):
    return Pipeline(
        [("scaler", RobustScaler()),
         ("model", GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=config.DATA_SPLIT["random_state"],
            ),
        )]
    )


@registry.experiment(name="hist_gradient_boost_tuned", desc="HistGradientBoosting with optimized hyperparameters")
def hist_gradient_boost_tuned(max_iter=300, max_depth=10, learning_rate=0.05, l2_regularization=0.1):
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        random_state=config.DATA_SPLIT["random_state"]
    )

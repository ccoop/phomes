import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import config
import pandas as pd
from evaluation import determine_best_model, evaluate_predictions, get_summary_metrics
from sklearn.pipeline import Pipeline


@dataclass
class Experiment:
    """Experiment container with all tracking information"""

    name: str
    pipeline: Pipeline
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    features_used: list[str] = field(default_factory=list)
    feature_count: int = 0
    data_version: str = field(default=None)
    id: str = field(init=False)
    created_at: datetime = field(init=False)

    def __post_init__(self):
        self.created_at = datetime.now()
        self.id = f"{self.name}_{self.created_at:%Y%m%d_%H%M%S}"


class ModelRegistry:
    """Manages experiment registry and model promotion"""

    def __init__(self):
        self.registry_path = Path(config.REGISTRY_FILE)
        self.experiments_dir = Path(config.REGISTRY_EXPERIMENTS_DIR)
        self.experiments: dict[str, Callable] = {}

    def load_registry(self) -> dict:
        """Load registry from file"""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"experiments": [], "best_model": None, "production_model": None}

    def save_registry(self, registry: dict) -> None:
        """Save registry to file"""
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def get_production_model(self) -> Optional[dict]:
        """Get current production model"""
        registry = self.load_registry()
        return registry.get("production_model")

    def get_experiment_metadata(self, experiment_id: str) -> dict:
        """Load experiment metadata from file"""
        metadata_path = self.experiments_dir / experiment_id / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Experiment {experiment_id} not found")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def evaluate_quality_gates(self, experiment_id: str) -> tuple[dict, bool]:
        """Evaluate if experiment meets quality gates"""
        exp_metadata = self.get_experiment_metadata(experiment_id)
        test_metrics = exp_metadata["metrics"]["test"]

        model_metrics = {
            "test_mape": test_metrics["mape"],
            "test_confidence_band_90pct": test_metrics["confidence_band_90pct"],
            "test_r2": test_metrics["r2"],
            "test_prediction_latency_ms": test_metrics.get("prediction_latency_ms", 0),
        }

        gates = config.QUALITY_GATES
        results = {
            "mape": model_metrics["test_mape"] <= gates["max_mape"],
            "confidence_band_90pct": model_metrics["test_confidence_band_90pct"] >= gates["min_confidence_band_90pct"],
            "r2": model_metrics["test_r2"] >= gates["min_r2"],
            "latency": model_metrics["test_prediction_latency_ms"] <= gates["max_latency_ms"],
        }

        current_prod = self.get_production_model()
        if current_prod:
            current_mape = current_prod["test_mape"]
            new_mape = model_metrics["test_mape"]
            improvement = (current_mape - new_mape) / current_mape * 100
            results["improvement"] = improvement >= gates["min_improvement_pct"]
        else:
            results["improvement"] = True

        all_passed = all(results.values()) if config.PROMOTION["require_all_gates"] else any(results.values())
        return results, all_passed

    def promote_to_production(self, experiment_id: str, force: bool = False) -> dict:
        """Promote experiment to production"""
        gate_results, gates_passed = self.evaluate_quality_gates(experiment_id)

        should_promote = gates_passed or (force and config.PROMOTION["allow_force_promote"])

        if should_promote:
            self._set_production_model(experiment_id)

        return {
            "experiment_id": experiment_id,
            "promoted": should_promote,
            "gates_passed": gates_passed,
            "gate_results": gate_results,
            "forced": force,
            "timestamp": datetime.now().isoformat(),
        }

    def _set_production_model(self, experiment_id: str) -> None:
        """Set experiment as production model"""
        exp_metadata = self.get_experiment_metadata(experiment_id)
        registry = self.load_registry()

        summary_metrics = get_summary_metrics(exp_metadata["metrics"])
        production_model = {
            "id": experiment_id,
            "name": exp_metadata["name"],
            "created_at": exp_metadata["created_at"],
            "promoted_at": datetime.now().isoformat(),
            "feature_count": exp_metadata["features"]["count"],
            "data_version": exp_metadata["data_version"],
            "fingerprint": exp_metadata["fingerprint"],
            **summary_metrics,
        }

        registry["production_model"] = production_model
        self.save_registry(registry)

    def add_experiment(self, experiment_summary: dict) -> None:
        """Add experiment to registry"""
        registry = self.load_registry()
        registry["experiments"].append(experiment_summary)
        registry["best_model"] = determine_best_model(registry.get("best_model"), experiment_summary)
        self.save_registry(registry)

    def find_existing_experiment(self, model_name: str, parameters: dict, data_version: str, features: list = None) -> dict | None:
        """Find if experiment already exists with same fingerprint"""
        fingerprint = self._calculate_fingerprint(model_name, parameters, data_version, features)
        registry = self.load_registry()

        for exp in registry.get("experiments", []):
            if exp.get("fingerprint") == fingerprint:
                if "test_mape" in exp and "test_confidence_band_90pct" in exp:
                    print(f"Found existing experiment with same fingerprint: {exp['id']}")
                    try:
                        return self.get_experiment_metadata(exp["id"])
                    except ValueError:
                        return exp
                else:
                    print(f"Found legacy experiment {exp['id']}, will retrain with new metrics")
                    return None

        return None

    def get_model(self, experiment_id: str) -> dict:
        """Get model pipeline and all metadata in one call"""
        exp_dir = Path(f"{config.REGISTRY_EXPERIMENTS_DIR}/{experiment_id}")

        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")

        model_path = exp_dir / "model.pkl"
        metadata_path = exp_dir / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            raise ValueError(f"Missing files for experiment {experiment_id}")

        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return {
            "pipeline": pipeline,
            "metadata": metadata,
            "features": metadata["features"]["names"],
            "metrics": metadata["metrics"]["test"],
        }

    def compare(self, exp_ids: list[str]):
        """Compare multiple experiments by loading their metadata"""
        from evaluation import compare_experiments_df

        comparison = []
        for exp_id in exp_ids:
            try:
                metadata = self.get_experiment_metadata(exp_id)
                comparison.append(metadata)
            except ValueError:
                continue

        return compare_experiments_df(comparison)

    def register_experiment(self, name: str, func: Callable, description: str = ""):
        """Register an experiment function"""
        wrapper_func = lambda **kwargs: Experiment(
            name=name, pipeline=func(**kwargs), description=description, parameters=kwargs
        )
        wrapper_func.metadata = {"name": name, "description": description, "func": func.__name__}
        self.experiments[name] = wrapper_func

    def list(self):
        """List all registered experiments"""
        for name, exp_func in self.experiments.items():
            meta = exp_func.metadata
            print(f"{name}: {meta['description']}")
            print()

    def _calculate_fingerprint(self, model_name: str, parameters: dict, data_version: str, features: list = None) -> str:
        """Create unique fingerprint for model+params+data+features combination"""
        param_str = json.dumps(parameters, sort_keys=True)
        feature_str = json.dumps(sorted(features or []), sort_keys=True)
        return hashlib.md5(f"{model_name}_{param_str}_{data_version}_{feature_str}".encode()).hexdigest()

    def experiment(self, name: str, desc: str = ""):
        """Decorator to register model experiments"""
        def decorator(func: Callable[..., Pipeline]):
            self.register_experiment(name, func, desc)
            return func
        return decorator

    def run_experiment(self, experiment: 'Experiment', X_train, y_train, X_val, y_val, X_test, y_test):
        """Execute experiment with full data tracking"""
        experiment.features_used = list(X_train.columns) if hasattr(X_train, "columns") else []
        experiment.feature_count = X_train.shape[1]

        # Temporarily disabled to test sample weights
        # existing = self.find_existing_experiment(
        #     experiment.name, experiment.parameters, experiment.data_version or "unknown", experiment.features_used
        # )
        # if existing:
        #     print(f"Skipping training - using existing experiment: {existing['id']}")
        #     return existing

        # Calculate MAPE-aligned sample weights (inverse price weighting)
        sample_weight = 1.0 / y_train
        sample_weight = sample_weight * len(y_train) / sample_weight.sum()
        
        # Apply weights - try with weights first, fallback without
        try:
            if hasattr(experiment.pipeline, 'named_steps'):
                # Pipeline with steps - pass to final model step
                experiment.pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
            else:
                # Direct model
                experiment.pipeline.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            experiment.pipeline.fit(X_train, y_train)

        y_train_pred = experiment.pipeline.predict(X_train)
        y_val_pred = experiment.pipeline.predict(X_val)
        y_test_pred = experiment.pipeline.predict(X_test)

        metrics = evaluate_predictions(
            y_train,
            y_train_pred,
            y_val,
            y_val_pred,
            y_test,
            y_test_pred,
            pipeline=experiment.pipeline,
            X_test=X_test,
        )

        fingerprint = self._calculate_fingerprint(
            experiment.name, experiment.parameters, experiment.data_version or "unknown", experiment.features_used
        )

        results = {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "created_at": experiment.created_at.isoformat(),
            "parameters": experiment.parameters,
            "fingerprint": fingerprint,
            "data_version": experiment.data_version,
            "features": {"names": experiment.features_used, "count": experiment.feature_count},
            "pipeline_steps": [f"{name}: {type(step).__name__}" for name, step in experiment.pipeline.steps] if hasattr(experiment.pipeline, 'steps') else [f"model: {type(experiment.pipeline).__name__}"],
            "hyperparameters": {k: str(v) for k, v in experiment.pipeline.get_params().items()},
            "metrics": metrics,
        }

        self._save_artifacts(experiment, results, X_test, y_test, y_test_pred)

        return results

    def _save_artifacts(self, experiment: 'Experiment', results, X_test, y_test, y_pred):
        """Save all experiment artifacts"""
        exp_dir = Path(f"{config.REGISTRY_EXPERIMENTS_DIR}/{experiment.id}")
        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / "model.pkl", "wb") as f:
            pickle.dump(experiment.pipeline, f)

        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(results, f, indent=2)

        predictions_df = pd.DataFrame({"actual": y_test, "predicted": y_pred, "residual": y_test - y_pred})
        predictions_df.to_csv(exp_dir / "predictions.csv", index=False)

        summary_metrics = get_summary_metrics(results["metrics"])
        summary = {
            "id": experiment.id,
            "name": experiment.name,
            "created_at": results["created_at"],
            "feature_count": experiment.feature_count,
            "data_version": experiment.data_version,
            "fingerprint": results["fingerprint"],
            **summary_metrics,
        }
        self.add_experiment(summary)




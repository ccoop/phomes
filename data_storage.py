"""
Data versioning and catalog management for ML.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config


class DataCatalog:
    """Manages dataset versions with flexible data loading and splitting."""

    def __init__(
        self,
        catalog_dir: str = None,
        data_loader: Optional[Callable] = None,
        split_config: Optional[Dict] = None,
    ):
        self.catalog_dir = Path(catalog_dir or config.DATA_CATALOG_DIR)
        self.versions_dir = self.catalog_dir / "versions"
        self.index_path = self.catalog_dir / "index.json"

        self.data_loader = data_loader or self._default_data_loader
        self.split_config = split_config or config.DATA_SPLIT

        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _default_data_loader(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Default data loading logic for housing data."""
        sales = pd.read_csv(config.DATA_SOURCES["sales_path"], dtype={"zipcode": str})
        demographics = pd.read_csv(config.DATA_SOURCES["demographics_path"], dtype={"zipcode": str})

        merged = sales.merge(demographics, how="left", on="zipcode")
        merged = merged.drop(columns="zipcode").dropna()

        y = merged.pop("price")
        X = merged.select_dtypes(include=[np.number])

        return X, y

    def _load_index(self):
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {"current": None, "next_id": 1, "versions": {}}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def snapshot(self, description: str = "") -> str:
        """Create a new data version snapshot."""
        version_id = f"v{self.index['next_id']}"

        X, y = self.data_loader()

        metadata = {
            "version": version_id,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "sources": self._detect_sources(),
            "split_config": self.split_config,
            "shape": {"samples": len(X), "features": X.shape[1]},
            "features": {
                "names": list(X.columns) if hasattr(X, "columns") else [],
                "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()}
                if hasattr(X, "dtypes")
                else {},
            },
            "target": {
                "name": "price",
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
            },
            "fingerprint": self._calculate_fingerprint(X, y),
        }

        version_path = self.versions_dir / f"{version_id}.json"
        with open(version_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.index["versions"][version_id] = {
            "created": metadata["created_at"],
            "samples": metadata["shape"]["samples"],
            "features": metadata["shape"]["features"],
            "description": description,
        }
        self.index["next_id"] += 1
        self.index["current"] = version_id
        self._save_index()

        return version_id

    def load_version(self, version_id: Optional[str] = None) -> Tuple:
        """Load data for a specific version with exact splits."""
        if version_id is None:
            if not self.index["current"]:
                version_id = self.snapshot("Initial dataset")
            else:
                version_id = self.index["current"]

        version = self.get_version_metadata(version_id)
        X, y = self.data_loader()

        # Warn if data has changed since version was created
        current_fingerprint = self._calculate_fingerprint(X, y)
        data_changed = current_fingerprint != version["fingerprint"]

        split_config = version["split_config"]

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=split_config["test_size"], random_state=split_config["random_state"]
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=split_config["val_size"], random_state=split_config["random_state"]
        )

        return X_train, y_train, X_val, y_val, X_test, y_test, version_id

    def list_versions(self) -> Dict:
        """Get all available data versions."""
        return self.index.get("versions", {})

    def get_version_metadata(self, version_id: str) -> Dict:
        """Get complete metadata for a version."""
        version_path = self.versions_dir / f"{version_id}.json"
        if not version_path.exists():
            raise ValueError(f"Version {version_id} not found")

        with open(version_path, "r") as f:
            return json.load(f)

    def get_current_version(self) -> Optional[str]:
        """Get the current version ID."""
        return self.index.get("current")

    def _detect_sources(self) -> Dict:
        """Detect data source files automatically."""
        sources = {}
        data_dir = Path(config.DATA_SOURCES["sales_path"]).parent

        for csv_file in data_dir.glob("*.csv"):
            with open(csv_file, "rb") as f:
                # Hash first 10MB for speed on large files
                content = f.read(10 * 1024 * 1024)
                file_hash = hashlib.sha256(content).hexdigest()[:16]

            sources[csv_file.stem] = {
                "path": str(csv_file),
                "hash": file_hash,
                "size": csv_file.stat().st_size,
                "modified": datetime.fromtimestamp(csv_file.stat().st_mtime).isoformat(),
            }

        return sources

    def _calculate_fingerprint(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Calculate a fingerprint for the dataset."""
        fingerprint_data = {
            "shape": X.shape,
            "columns": list(X.columns) if hasattr(X, "columns") else [],
            "target_stats": {"mean": float(y.mean()), "std": float(y.std())},
            "sample_hash": hashlib.md5(
                X.head(100).to_string().encode() if hasattr(X, "head") else b""
            ).hexdigest(),
        }

        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]


def load_housing_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Standard housing data loader for backward compatibility."""
    sales = pd.read_csv(config.DATA_SOURCES["sales_path"], dtype={"zipcode": str})
    demographics = pd.read_csv(config.DATA_SOURCES["demographics_path"], dtype={"zipcode": str})

    merged = sales.merge(demographics, how="left", on="zipcode")
    merged = merged.drop(columns="zipcode").dropna()

    y = merged.pop("price")
    X = merged.select_dtypes(include=[np.number])

    return X, y

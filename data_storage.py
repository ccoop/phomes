import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import config
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class HomeSale(BaseModel):
    """Pydantic model for validating house sale data."""

    id: int = Field(gt=0, description="Unique house ID")
    date: str = Field(description="Sale date")
    price: float = Field(gt=0, lt=50_000_000, description="Sale price")
    bedrooms: int = Field(ge=0, description="Number of bedrooms")
    bathrooms: float = Field(ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(gt=0, le=50_000, description="Living space square footage")
    sqft_lot: int = Field(gt=0, description="Lot square footage")
    floors: float = Field(ge=1, le=10, description="Number of floors")
    waterfront: int = Field(ge=0, le=1, description="Waterfront property (0 or 1)")
    view: int = Field(ge=0, le=4, description="View rating")
    condition: int = Field(ge=1, le=5, description="Overall condition")
    grade: int = Field(ge=1, le=13, description="Overall grade")
    sqft_above: int = Field(ge=0, le=50_000, description="Above ground square footage")
    sqft_basement: int = Field(ge=0, le=50_000, description="Basement square footage")
    yr_built: int = Field(ge=1800, le=2025, description="Year built")
    yr_renovated: int = Field(ge=0, le=2025, description="Year renovated")
    zipcode: str = Field(min_length=5, max_length=5, description="Zipcode")
    lat: float = Field(ge=47.0, le=48.0, description="Latitude")
    long: float = Field(ge=-123.0, le=-121.0, description="Longitude")
    sqft_living15: int = Field(ge=0, le=50_000, description="Avg living space of 15 nearest neighbors")
    sqft_lot15: int = Field(ge=0, le=1_000_000, description="Avg lot size of 15 nearest neighbors")

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date is in expected format."""
        if not v or len(v) < 8:
            raise ValueError("Date must be in YYYYMMDT format")
        return v


class ZipcodeDemographics(BaseModel):
    """Pydantic model for validating zipcode demographic data."""

    ppltn_qty: int = Field(ge=0, description="Total population")
    urbn_ppltn_qty: int = Field(ge=0, description="Urban population")
    sbrbn_ppltn_qty: int = Field(ge=0, description="Suburban population")
    farm_ppltn_qty: int = Field(ge=0, description="Farm population")
    non_farm_qty: int = Field(ge=0, description="Non-farm population")
    medn_hshld_incm_amt: float = Field(ge=0, description="Median household income")
    medn_incm_per_prsn_amt: float = Field(ge=0, description="Median income per person")
    hous_val_amt: float = Field(ge=0, description="Housing value amount")
    edctn_less_than_9_qty: int = Field(ge=0, description="Education less than 9th grade")
    edctn_9_12_qty: int = Field(ge=0, description="Education 9-12 grade")
    edctn_high_schl_qty: int = Field(ge=0, description="High school education")
    edctn_some_clg_qty: int = Field(ge=0, description="Some college education")
    edctn_assoc_dgre_qty: int = Field(ge=0, description="Associate degree")
    edctn_bchlr_dgre_qty: int = Field(ge=0, description="Bachelor's degree")
    edctn_prfsnl_qty: int = Field(ge=0, description="Professional degree")
    per_urbn: float = Field(ge=0, le=100, description="Percent urban")
    per_sbrbn: float = Field(ge=0, le=100, description="Percent suburban")
    per_farm: float = Field(ge=0, le=100, description="Percent farm")
    per_non_farm: float = Field(ge=0, le=100, description="Percent non-farm")
    per_less_than_9: float = Field(ge=0, le=100, description="Percent less than 9th grade")
    per_9_to_12: float = Field(ge=0, le=100, description="Percent 9-12 grade")
    per_hsd: float = Field(ge=0, le=100, description="Percent high school")
    per_some_clg: float = Field(ge=0, le=100, description="Percent some college")
    per_assoc: float = Field(ge=0, le=100, description="Percent associate degree")
    per_bchlr: float = Field(ge=0, le=100, description="Percent bachelor's degree")
    per_prfsnl: float = Field(ge=0, le=100, description="Percent professional degree")
    zipcode: str = Field(min_length=5, max_length=5, description="Zipcode")

    @field_validator('ppltn_qty', 'urbn_ppltn_qty', 'sbrbn_ppltn_qty', 'farm_ppltn_qty', 'non_farm_qty',
                     'edctn_less_than_9_qty', 'edctn_9_12_qty', 'edctn_high_schl_qty', 'edctn_some_clg_qty',
                     'edctn_assoc_dgre_qty', 'edctn_bchlr_dgre_qty', 'edctn_prfsnl_qty', mode='before')
    @classmethod
    def round_counts(cls, v):
        """Round fractional population/education counts to integers."""
        return int(round(float(v))) if v is not None else v


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

    def _validate_dataframe(self, df: pd.DataFrame, model_class: type) -> pd.DataFrame:
        """Validate DataFrame rows using Pydantic model."""
        validated_records = []
        invalid_count = 0

        for idx, row in df.iterrows():
            try:
                validated_record = model_class(**row.to_dict())
                validated_records.append(validated_record.dict())
            except Exception as e:
                logger.warning(f"Invalid {model_class.__name__} record at row {idx}: {e}")
                invalid_count += 1

        logger.info(f"Validated {len(validated_records)} {model_class.__name__} records, rejected {invalid_count}")
        return pd.DataFrame(validated_records)

    def _default_data_loader(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Default data loading logic for housing data with Pydantic validation."""
        sales = self.load_source("sales", validate=True)
        demographics = self.load_source("demographics", validate=True)

        merged = sales.merge(demographics, how="left", on="zipcode")
        merged = merged.drop(columns="zipcode")
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].median())

        y = merged.pop("price")
        X = merged.select_dtypes(include=[np.number])

        # Remove id field - it's not a useful feature
        if "id" in X.columns:
            X = X.drop(columns=["id"])

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
        split_config = version["split_config"]
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"]
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=split_config["val_size"],
            random_state=split_config["random_state"]
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

    def load_source(self, source_name: str, validate: bool = True) -> pd.DataFrame:
        """Load and optionally validate a data source by name.

        Args:
            source_name: Name of source ('sales', 'demographics', 'future_unseen_examples')
            validate: Whether to apply Pydantic validation

        Returns:
            Validated DataFrame with consistent dtypes
        """
        # Map source names to file paths and validation models
        source_config = {
            "sales": {"path_key": "sales_path", "model": HomeSale},
            "demographics": {"path_key": "demographics_path", "model": ZipcodeDemographics},
            "future_unseen_examples": {"path_key": "future_unseen_examples_path", "model": None}
        }

        if source_name not in source_config:
            raise ValueError(f"Unknown source: {source_name}. Available: {list(source_config.keys())}")

        config_info = source_config[source_name]
        file_path = config.DATA_SOURCES[config_info["path_key"]]

        # Load with consistent dtypes (zipcode as string)
        df = pd.read_csv(file_path, dtype={"zipcode": str})

        # Apply validation if requested and model available
        if validate and config_info["model"]:
            df = self._validate_dataframe(df, config_info["model"])

        return df

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


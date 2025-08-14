#!/usr/bin/env python3
"""
CLI interface for the ML experiment system.
"""

import argparse
from pathlib import Path

import models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config
from shared import catalog, registry




def parse_params(param_strings):
    """Parse parameter strings in format key=value"""
    params = {}
    if not param_strings:
        return params

    for param in param_strings:
        if "=" not in param:
            print(f"Warning: Invalid parameter format '{param}', should be key=value")
            continue

        key, value = param.split("=", 1)

        try:
            if "." not in value:
                params[key] = int(value)
            else:
                params[key] = float(value)
        except ValueError:
            params[key] = value

    return params


def train_cmd(args):
    """Handle the train command"""
    # Validate arguments
    if not args.all and not args.model:
        print("Error: Must specify either a model name or --all")
        print("Available models:", list(registry.experiments.keys()))
        return
    
    # Determine which models to train
    if args.all:
        models_to_train = list(registry.experiments.keys())
        print(f"Training all {len(models_to_train)} models: {', '.join(models_to_train)}")
    else:
        if args.model not in registry.experiments:
            print(f"Unknown model: {args.model}")
            print("Available models:", list(registry.experiments.keys()))
            return
        models_to_train = [args.model]

    params = parse_params(args.params)
    data_version = getattr(args, 'data', None)
    
    # Load data once
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, version_id = \
        catalog.load_version(data_version)

    print(f"Data loaded: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)} samples")
    print(f"Features: {X_train.shape[1]} | Data version: {version_id}")

    # Train each model
    for i, model_name in enumerate(models_to_train, 1):
        if len(models_to_train) > 1:
            print(f"\n{'=' * 60}")
            print(f"Training model {i}/{len(models_to_train)}: {model_name}")
            print(f"{'=' * 60}")
        
        print(f"Creating experiment: {model_name}")
        if params:
            print(f"Parameters: {params}")

        exp = registry.experiments[model_name](**params)
        exp.data_version = version_id

        if args.note:
            exp.description = f"{exp.description} - {args.note}"

        print(f"Running experiment: {exp.id}")
        print("Training model...")

        results = registry.run_experiment(exp, X_train, y_train, X_val, y_val, X_test, y_test)

        print(f"\nExperiment Complete: {results['id']}")
        print(f"Description: {results['description']}")

        from eval import format_metrics_for_display
        print(f"\n{format_metrics_for_display(results['metrics'])}")

        print(f"\n💾 Data:")
        print(f"  Features: {results['features']['count']}")
        print(f"  Data version: {version_id}")
        print(f"Artifacts saved to: {config.REGISTRY_EXPERIMENTS_DIR}/{results['id']}/")

    if len(models_to_train) > 1:
        print(f"\n✅ Completed training {len(models_to_train)} models. Use 'mla registry' to view results.")


def serve_cmd(args):
    """Handle the serve command"""
    import subprocess
    import sys
    
    host = getattr(args, 'host', '0.0.0.0')
    port = getattr(args, 'port', 8000)
    
    print(f"Starting API server on {host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "api:app", 
            "--host", host, "--port", str(port), "--reload"
        ])
    except KeyboardInterrupt:
        print("\nServer stopped")


def data_cmd(args):
    """Handle data commands"""
    match args.data_command:
        case "snapshot":
            note = getattr(args, 'note', '')
            version_id = catalog.snapshot(note)
            
            versions = catalog.list_versions()
            meta = versions[version_id]
            print(f"Created data version: {version_id}")
            print(f"  Samples: {meta['samples']:,}")
            print(f"  Features: {meta['features']}")
            if note:
                print(f"  Description: {note}")
        
        case "list":
            versions = catalog.list_versions()
            
            if not versions:
                print("No data versions found. Use 'data snapshot' to create one.")
                return
            
            current = catalog.get_current_version()
            
            print("Data Versions:")
            print("-" * 60)
            
            for vid in sorted(versions.keys()):
                meta = versions[vid]
                is_current = " (current)" if vid == current else ""
                created = meta["created"].split("T")[0]
                
                print(f"{vid}{is_current}")
                print(f"  Created: {created}")
                if meta.get("description"):
                    print(f"  Description: {meta['description']}")
                print(f"  Samples: {meta['samples']:,} | Features: {meta['features']}")
                print()


def list_cmd(args):
    """Handle the list command"""
    print("Available Experiments:")
    print("=" * 50)
    registry.list()


def compare_cmd(args):
    """Handle the compare command"""
    if len(args.experiments) < 2:
        print("Please provide at least 2 experiment IDs to compare")
        return

    print(f"Comparing experiments: {', '.join(args.experiments)}")

    try:
        df = registry.compare(args.experiments)
        print("\nComparison:")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Error comparing experiments: {e}")


def promote_cmd(args):
    """Handle the promote command"""
    try:
        result = registry.promote_to_production(args.experiment_id, force=args.force)
        
        print(f"Promotion Report for {args.experiment_id}:")
        print("=" * 50)
        
        if result["promoted"]:
            print("✅ Model promoted to production!")
            if result["forced"]:
                print("⚠️  Promotion was forced (quality gates may have failed)")
        else:
            print("❌ Promotion failed - quality gates not met")
        
        print(f"\nQuality Gates:")
        for gate, passed in result["gate_results"].items():
            status = "✅" if passed else "❌"
            print(f"  {gate}: {status}")
        
        if not result["gates_passed"] and not args.force:
            print(f"\nUse --force to override quality gates")
        
        print(f"\nTimestamp: {result['timestamp']}")
        
    except ValueError as e:
        print(f"Error: {e}")


def registry_cmd(args):
    """Handle the registry command"""
    registry_data = registry.load_registry()

    if not registry_data["experiments"]:
        print("No experiments found. Run some experiments first!")
        return

    print("Experiment Registry:")
    print("=" * 50)
    print(f"Total experiments: {len(registry_data['experiments'])}")

    if registry_data.get("production_model"):
        prod = registry_data["production_model"]
        mape = prod.get("test_mape", "N/A")
        acc_15 = prod.get("test_accuracy_15pct", "N/A")
        promoted_date = prod.get("promoted_at", "").split("T")[0]
        print(f"Production model: {prod['id']} (promoted {promoted_date})")
        if isinstance(mape, (int, float)):
            print(f"  Performance: {mape:.1f}% MAPE, {acc_15:.1f}% within 15%")

    if registry_data["best_model"]:
        best = registry_data["best_model"]
        mape = best.get("test_mape", "N/A")
        acc_15 = best.get("test_accuracy_15pct", "N/A")
        if isinstance(mape, (int, float)):
            print(f"Best model: {best['id']} ({mape:.1f}% MAPE, {acc_15:.1f}% within 15%)")

    print(f"\n{'Rank':<4} | {'Experiment':<35} | {'Prod':<4} | {'Data':<4} | {'Date':<10} | {'MAPE':<6} | {'15% Acc':<7} | {'MAE':<8}")
    print("-" * 88)

    def sort_key(exp):
        if "test_mape" in exp and exp["test_mape"] is not None:
            return exp["test_mape"]
        elif "test_mae" in exp and exp["test_mae"] is not None:
            return exp["test_mae"]
        return exp.get("test_rmse", float("inf"))

    production_id = registry_data.get("production_model", {}).get("id")
    best_id = registry_data.get("best_model", {}).get("id")

    for i, exp in enumerate(sorted(registry_data["experiments"], key=sort_key), 1):
        date = exp["created_at"].split("T")[0]
        
        if exp["id"] == production_id:
            rank = "🚀"
        elif exp["id"] == best_id:
            rank = "⭐"
        else:
            rank = f"{i:2d}"
        
        prod_status = "🚀" if exp["id"] == production_id else ""
        
        mape_str = f"{exp.get('test_mape', 'N/A'):.1f}%" if exp.get("test_mape") is not None else "N/A"
        acc_str = (
            f"{exp.get('test_accuracy_15pct', 'N/A'):.1f}%"
            if exp.get("test_accuracy_15pct") is not None
            else "N/A"
        )

        mae_value = exp.get("test_mae", exp.get("test_rmse", 0))
        mae_str = f"${mae_value:,.0f}"
        
        data_v = exp.get('data_version', '-')

        print(f"{rank:<4} | {exp['id']:<35} | {prod_status:<4} | {data_v:<4} | {date} | {mape_str:<6} | {acc_str:<7} | {mae_str:<8}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="ML Experiment CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("model", nargs="?", help="Model name to train")
    train_parser.add_argument("--all", action="store_true", help="Train all available models")
    train_parser.add_argument("--params", nargs="+", help="Override params (key=value format)")
    train_parser.add_argument("--note", help="Note about this experiment")
    train_parser.add_argument("--data", help="Specific data version to use (e.g., v1)")

    list_parser = subparsers.add_parser("list", help="List available models")

    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiments", nargs="+", help="Experiment IDs to compare")

    registry_parser = subparsers.add_parser("registry", help="Show experiment registry")

    promote_parser = subparsers.add_parser("promote", help="Promote model to production")
    promote_parser.add_argument("experiment_id", help="Experiment ID to promote")
    promote_parser.add_argument("--force", action="store_true", help="Force promotion despite quality gate failures")

    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")

    data_parser = subparsers.add_parser("data", help="Data version management")
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data commands")
    
    snapshot_parser = data_subparsers.add_parser("snapshot", help="Create data snapshot")
    snapshot_parser.add_argument("--note", help="Description for this version")
    
    list_data_parser = data_subparsers.add_parser("list", help="List data versions")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    match args.command:
        case "train":
            train_cmd(args)
        case "list":
            list_cmd(args)
        case "compare":
            compare_cmd(args)
        case "registry":
            registry_cmd(args)
        case "promote":
            promote_cmd(args)
        case "serve":
            serve_cmd(args)
        case "data":
            data_cmd(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse

import config
from shared import catalog, registry
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


def parse_params(param_strings):
    """Parse parameter strings in format key=value"""
    params = {}
    if not param_strings:
        return params

    for param in param_strings:
        if "=" not in param:
            console.print(f"[yellow]Warning:[/yellow] Invalid parameter format '{param}', should be key=value")
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


def filter_features(*datasets, include=None, exclude=None):
    """Apply consistent feature filtering to datasets"""
    if include and exclude:
        raise ValueError("Cannot use both include and exclude")

    if include:
        return [df[include] for df in datasets]
    elif exclude:
        return [df.drop(columns=exclude, errors='ignore') for df in datasets]
    else:
        return datasets


def train_cmd(args):
    """Handle the train command"""

    if not args.all and not args.model:
        console.print("[red]Error:[/red] Must specify either a model name or --all")
        console.print("Available models:", list(registry.experiments.keys()))
        return

    # Determine which models to train
    if args.all:
        models_to_train = list(registry.experiments.keys())
        console.print(f"Training all {len(models_to_train)} models: {', '.join(models_to_train)}")
    else:
        if args.model not in registry.experiments:
            console.print(f"[red]Unknown model:[/red] {args.model}")
            console.print("Available models:", list(registry.experiments.keys()))
            return
        models_to_train = [args.model]

    params = parse_params(args.params)
    data_version = getattr(args, 'data', None)

    console.print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, version_id = \
        catalog.load_version(data_version)

    # Apply feature selection if specified
    include_features = args.include.split(',') if args.include else None
    exclude_features = args.exclude.split(',') if args.exclude else None

    X_train, X_val, X_test = filter_features(X_train, X_val, X_test,
                                           include=include_features,
                                           exclude=exclude_features)

    console.print(f"Data loaded: Train {len(X_train):,}, Val {len(X_val):,}, Test {len(X_test):,} samples")
    console.print(f"Features: {X_train.shape[1]} | Data version: {version_id}")

    # Show feature filtering details
    if exclude_features:
        console.print(f"[yellow]Excluded features:[/yellow] {', '.join(exclude_features)}")
    if include_features:
        console.print(f"[yellow]Included features:[/yellow] {', '.join(include_features)}")

    # Use progress tracking if training multiple models
    if len(models_to_train) > 1:
        models_iter = track(models_to_train, description="Training models...")
    else:
        models_iter = models_to_train

    for model_name in models_iter:
        if len(models_to_train) > 1:
            console.print(f"\n[bold]Training: {model_name}[/bold]")

        console.print(f"Creating experiment: {model_name}")
        if params:
            console.print(f"Parameters: {params}")

        exp = registry.experiments[model_name](**params)
        exp.data_version = version_id

        if args.note:
            exp.description = f"{exp.description} - {args.note}"

        console.print(f"Running experiment: {exp.id}")
        console.print("Training model...")

        results = registry.run_experiment(exp, X_train, y_train, X_val, y_val, X_test, y_test)

        console.print(f"\n[bold]Experiment Complete:[/bold] {results['id']}")
        console.print(f"Description: {results['description']}")

        test_metrics = results['metrics']['test']
        mape = test_metrics.get('mape')
        acc15 = test_metrics.get('accuracy_within_15pct')
        mae = test_metrics.get('mae')
        r2 = test_metrics.get('r2')

        console.print(f"\n[bold]Performance:[/bold]")
        console.print(f"  MAPE: {mape:.1f}%" if mape is not None else "  MAPE: N/A")
        console.print(f"  Accuracy within 15%: {acc15:.1f}%" if acc15 is not None else "  Accuracy within 15%: N/A")
        console.print(f"  MAE: ${mae:,.0f}" if mae is not None else "  MAE: N/A")
        console.print(f"  R²: {r2:.3f}" if r2 is not None else "  R²: N/A")

        console.print(f"\n[bold]Data:[/bold]")
        console.print(f"  Features: {results['features']['count']}")
        console.print(f"  Data version: {version_id}")
        console.print(f"Artifacts saved to: {config.REGISTRY_EXPERIMENTS_DIR}/{results['id']}/")

    if len(models_to_train) > 1:
        console.print(f"\n[green]Completed training {len(models_to_train)} models. Use 'mla models list' to view results.[/green]")


def serve_cmd(args):
    """Handle the serve command"""
    import subprocess
    import sys

    host = getattr(args, 'host', '0.0.0.0')
    port = getattr(args, 'port', 8000)

    console.print(f"Starting API server on {host}:{port}")
    console.print("Press Ctrl+C to stop the server")

    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "api:app",
            "--host", host, "--port", str(port), "--reload"
        ])
    except KeyboardInterrupt:
        console.print("\nServer stopped")


def data_cmd(args):
    """Handle data commands"""
    match args.data_command:
        case "snapshot":
            note = getattr(args, 'note', '')
            version_id = catalog.snapshot(note)

            versions = catalog.list_versions()
            meta = versions[version_id]
            console.print(f"[green]Created data version:[/green] {version_id}")
            console.print(f"  Samples: {meta['samples']:,}")
            console.print(f"  Features: {meta['features']}")
            if note:
                console.print(f"  Description: {note}")

        case "list":
            versions = catalog.list_versions()

            if not versions:
                console.print("No data versions found. Use 'data snapshot' to create one.")
                return

            current = catalog.get_current_version()

            console.print(f"\n[bold]Data Versions[/bold]")

            table = Table(show_header=True, header_style="bold")
            table.add_column("Version", width=15)
            table.add_column("Created", width=12)
            table.add_column("Description", width=30)
            table.add_column("Samples", width=10, justify="right")
            table.add_column("Features", width=8, justify="right")

            for vid in sorted(versions.keys()):
                meta = versions[vid]
                is_current = vid == current
                created = meta["created"].split("T")[0]
                description = meta.get("description", "")

                version_str = f"[bold]{vid}[/bold] (current)" if is_current else vid
                samples_str = f"{meta['samples']:,}"
                features_str = str(meta['features'])

                table.add_row(version_str, created, description, samples_str, features_str)

            console.print(table)


def models_cmd(args):
    """Handle the models command"""
    match args.models_command:
        case "list":
            models_list_cmd(args)
        case "show":
            models_show_cmd(args)
        case "compare":
            models_compare_cmd(args)
        case "promote":
            models_promote_cmd(args)


def models_list_cmd(args):
    """Handle the models list command"""
    registry_data = registry.load_registry()

    if not registry_data["experiments"]:
        console.print("No experiments found. Run some experiments first!")
        return

    console.print(f"\n[bold]Experiment Registry[/bold]")
    console.print(f"Total experiments: {len(registry_data['experiments'])}")

    if registry_data.get("production_model"):
        prod = registry_data["production_model"]
        mape = prod.get("test_mape", "N/A")
        conf_band = prod.get("test_confidence_band_90pct", "N/A")
        promoted_date = prod.get("promoted_at", "").split("T")[0]
        console.print(f"Production model: {prod['id']} (promoted {promoted_date})")
        if isinstance(mape, (int, float)) and isinstance(conf_band, (int, float)):
            console.print(f"  Performance: {mape:.1f}% MAPE, {conf_band/10:.1f}/10 confidence band")

    if registry_data["best_model"]:
        best = registry_data["best_model"]
        mape = best.get("test_mape", "N/A")
        conf_band = best.get("test_confidence_band_90pct", "N/A")
        if isinstance(mape, (int, float)) and isinstance(conf_band, (int, float)):
            console.print(f"Best model: {best['id']} ({mape:.1f}% MAPE, {conf_band/10:.1f}/10 confidence band)")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Rank", width=4)
    table.add_column("Experiment", width=50)
    table.add_column("Data", width=4)
    table.add_column("Features", width=8, justify="right")
    table.add_column("MAPE", width=6, justify="right")
    table.add_column("Conf Band", width=9, justify="right")
    table.add_column("MAE", width=8, justify="right")

    def sort_key(exp):
        if "test_mape" in exp and exp["test_mape"] is not None:
            return exp["test_mape"]
        elif "test_mae" in exp and exp["test_mae"] is not None:
            return exp["test_mae"]
        return exp.get("test_rmse", float("inf"))

    production_id = registry_data.get("production_model", {}).get("id") if registry_data.get("production_model") else None
    best_id = registry_data.get("best_model", {}).get("id") if registry_data.get("best_model") else None

    for i, exp in enumerate(sorted(registry_data["experiments"], key=sort_key), 1):
        if exp["id"] == production_id:
            rank = "PROD"
        elif exp["id"] == best_id:
            rank = "BEST"
        else:
            rank = f"{i:2d}"

        mape_str = f"{exp.get('test_mape', 'N/A'):.1f}%" if exp.get("test_mape") is not None else "N/A"
        conf_str = (
            f"{exp.get('test_confidence_band_90pct', 'N/A')/10:.1f}/10"
            if exp.get("test_confidence_band_90pct") is not None
            else "N/A"
        )

        mae_value = exp.get("test_mae", exp.get("test_rmse", 0))
        mae_str = f"${mae_value:,.0f}" if mae_value else "N/A"
        data_v = exp.get('data_version', '-')
        features_count = exp.get('feature_count', exp.get('features', {}).get('count', 'N/A'))

        table.add_row(rank, exp['id'], data_v, str(features_count), mape_str, conf_str, mae_str)

    console.print(table)


def models_show_cmd(args):
    """Handle the models show command"""
    target = args.target

    registry_data = registry.load_registry()

    if target == "prod":
        if not registry_data.get("production_model"):
            console.print("No production model set")
            return
        experiment_id = registry_data["production_model"]["id"]
    elif target == "best":
        if not registry_data.get("best_model"):
            console.print("No best model found")
            return
        experiment_id = registry_data["best_model"]["id"]
    else:
        experiment_id = target

    try:
        metadata = registry.get_experiment_metadata(experiment_id)

        console.print(f"\n[bold]Experiment Details: {experiment_id}[/bold]")
        console.print(f"Name: {metadata['name']}")
        console.print(f"Description: {metadata.get('description', 'N/A')}")
        console.print(f"Created: {metadata['created_at'].split('T')[0]}")
        console.print(f"Data Version: {metadata.get('data_version', 'N/A')}")
        console.print(f"Features: {metadata['features']['count']}")

        if metadata.get('parameters'):
            console.print(f"\n[bold]Parameters:[/bold]")
            for key, value in metadata['parameters'].items():
                console.print(f"  {key}: {value}")

        console.print(f"\n[bold]Metrics:[/bold]")
        test_metrics = metadata['metrics']['test']
        console.print(f"  MAPE: {test_metrics.get('mape', 'N/A'):.1f}%" if test_metrics.get('mape') is not None else "  MAPE: N/A")
        console.print(f"  Accuracy within 15%: {test_metrics.get('accuracy_within_15pct', 'N/A'):.1f}%" if test_metrics.get('accuracy_within_15pct') is not None else "  Accuracy within 15%: N/A")
        console.print(f"  MAE: ${test_metrics.get('mae', 'N/A'):,.0f}" if test_metrics.get('mae') is not None else "  MAE: N/A")
        console.print(f"  R²: {test_metrics.get('r2', 'N/A'):.3f}" if test_metrics.get('r2') is not None else "  R²: N/A")

        if test_metrics.get('prediction_latency_ms') is not None:
            console.print(f"  Prediction Latency: {test_metrics['prediction_latency_ms']:.2f}ms")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")


def models_compare_cmd(args):
    """Handle the models compare command"""
    if len(args.experiments) < 2:
        console.print("Please provide at least 2 experiment IDs to compare")
        return

    console.print(f"Comparing experiments: {', '.join(args.experiments)}")

    try:
        df = registry.compare(args.experiments)

        table = Table(show_header=True, header_style="bold")

        # Add columns with focused, readable formatting
        table.add_column("Experiment", width=45)
        table.add_column("MAPE %", width=8, justify="right")
        table.add_column("Conf Band", width=9, justify="right")
        table.add_column("MAE $", width=10, justify="right")
        table.add_column("R²", width=6, justify="right")
        table.add_column("Features", width=8, justify="right")
        table.add_column("Data", width=6)

        for _, row in df.iterrows():
            exp_id = str(row['id'])
            mape = f"{row['test_mape']:.1f}" if isinstance(row['test_mape'], (int, float)) else str(row['test_mape'])
            conf_band = f"{row['confidence_band_90pct']/10:.1f}/10" if isinstance(row['confidence_band_90pct'], (int, float)) else str(row['confidence_band_90pct'])
            mae = f"${row['test_mae']:,.0f}" if isinstance(row['test_mae'], (int, float)) else str(row['test_mae'])
            r2 = f"{row['test_r2']:.3f}" if isinstance(row['test_r2'], (int, float)) else str(row['test_r2'])
            features = str(row['features'])
            data_ver = str(row['data_version'])

            table.add_row(exp_id, mape, conf_band, mae, r2, features, data_ver)

        console.print("\n")
        console.print(table)
    except Exception as e:
        console.print(f"Error comparing experiments: {e}")


def models_promote_cmd(args):
    """Handle the models promote command"""
    try:
        result = registry.promote_to_production(args.experiment_id, force=args.force)

        console.print(f"\n[bold]Promotion Report for {args.experiment_id}:[/bold]")

        if result["promoted"]:
            console.print("[green]Model promoted to production![/green]")
            if result["forced"]:
                console.print("[yellow]Promotion was forced (quality gates may have failed)[/yellow]")
        else:
            console.print("[red]Promotion failed - quality gates not met[/red]")

        console.print("\n[bold]Quality Gates:[/bold]")
        for gate, passed in result["gate_results"].items():
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            console.print(f"  {gate}: {status}")

        if not result["gates_passed"] and not args.force:
            console.print("\nUse --force to override quality gates")

        console.print(f"\nTimestamp: {result['timestamp']}")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")




def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Machine Learning CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("model", nargs="?", help="Model name to train")
    train_parser.add_argument("--all", action="store_true", help="Train all available models")
    train_parser.add_argument("--params", nargs="+", help="Override params (key=value format)")
    train_parser.add_argument("--note", help="Note about this experiment")
    train_parser.add_argument("--data", help="Specific data version to use (e.g., v1)")
    train_parser.add_argument("--exclude", help="Comma-separated list of features to exclude")
    train_parser.add_argument("--include", help="Comma-separated list of features to include (mutually exclusive with --exclude)")

    models_parser = subparsers.add_parser("models", help="Model management commands")
    models_subparsers = models_parser.add_subparsers(dest="models_command", help="Models commands")

    models_list_parser = models_subparsers.add_parser("list", help="List all experiments")

    models_show_parser = models_subparsers.add_parser("show", help="Show experiment details")
    models_show_parser.add_argument("target", help="Experiment ID, 'prod', or 'best'")

    models_compare_parser = models_subparsers.add_parser("compare", help="Compare experiments")
    models_compare_parser.add_argument("experiments", nargs="+", help="Experiment IDs to compare")

    models_promote_parser = models_subparsers.add_parser("promote", help="Promote model to production")
    models_promote_parser.add_argument("experiment_id", help="Experiment ID to promote")
    models_promote_parser.add_argument("--force", action="store_true", help="Force promotion despite quality gate failures")

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
        case "models":
            models_cmd(args)
        case "serve":
            serve_cmd(args)
        case "data":
            data_cmd(args)


if __name__ == "__main__":
    main()

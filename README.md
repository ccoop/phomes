# phData ML Engineer Project - Usage Guide
My goal for the project was to implement a toolkit for ML workflows
that would help a data scientist iterate quickly and get their models ready
for deployment. I thought it would be more interesting to make this vendor-neutral
so this is all pure Python without external SAAS services or cloud providers.

If you are running the container interactively using -it, here's how to
get an overview of the project:

## Quick Start for Reviewing

```bash
docker build -t phdata-homes .
docker run -it --rm phdata-homes
```
The codebase uses a cli to simplify runnning common ml workflows like
managing the model registry, data catalog, and api server.

If you are running the container interactively using -it, here's how to
get an overview of the project:

```bash
# Review model history in the registry
mla models list

# Run API tests including sample requests
pytest tests/test_api.py -v

# To start the API server
mla serve &

```

## Overview of CLI Commands (use --help for more details)

```bash
mla models list          # All experiments ranked by performance
mla models show best     # Best model details
mla data list           # Data versions
mla train knn_baseline  # Train new model (if that ver. not already in registry)
mla serve               # Start API server
```

## Alternative: Direct Commands

```bash
# Without entering container
docker run --rm phdata-homes bash -c "mla models list"
docker run -d -p 8000:8000 phdata-homes bash -c "mla serve"
```
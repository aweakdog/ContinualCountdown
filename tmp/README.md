# Countdown Data Viewer

This directory contains tools for viewing and exploring the Countdown task dataset. The main script `show_data.py` provides an interactive interface to browse through the dataset examples.

## Docker Setup

### 1. Build the Docker Image

```bash
# From the project root directory
docker compose build countdown-viewer
```

### 2. Run the Container

```bash
# Run in interactive mode
docker compose run --rm countdown-viewer
```

## Interactive Commands

Once the viewer is running, you can use these commands:
- Enter a number N to view next N examples
- Type 'jump' to jump to a specific index
- Type 'stats' to see dataset statistics
- Type 'exit' to quit
- Use Ctrl+C to exit gracefully

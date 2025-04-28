# B-Spline Fitting for 3D Plant Point Clouds

## Overview

This script takes preprocessed 3D plant point clouds (`.npy` files) and turns them into smooth, parametric branch models using cubic B-splines. The main steps include:

1. **Skeleton Extraction**: Shrink and connect points to form a simplified plant skeleton.  
2. **Branch Detection**: Identify branch paths from leaf tips back to the root node.  
3. **Spline Fitting**: Fit a cubic B-spline (4 control points by default) to each branch, with optional emphasis on horizontal (x/y) coordinates.  
4. **Visualization**: Produce 3D plots showing the original point cloud, the skeleton, and the fitted splines.  
5. **Export**: Save branch parameters in JSON and output rendering images.

The pipeline can process multiple plants in batch mode and is configurable via command-line flags.

## Features

- **Point Cloud Shrinking**: Merge nearby points and filter edges to reduce noise and complexity.  
- **Graph Simplification**: Ensure connectivity using a minimum spanning tree and K-means clustering.  
- **Weighted Spline Fitting**: Use `scipy.splprep` with custom weights to improve fit in regions of high horizontal variation.  
- **Configurable Control Points**: Change how many spline control points by adjusting a function parameter.  
- **Batch Processing**: Automatically process all files in the specified input directory.  
- **Detailed Logging**: Supports DEBUG/INFO/WARNING/ERROR levels for tracking each pipeline stage.

## Requirements

- Python 3.7 or later  
- NumPy  
- SciPy  
- Matplotlib  
- NetworkX  
- scikit-learn  

Install dependencies via:

```bash
pip install numpy scipy matplotlib networkx scikit-learn
```

## Project Layout

```
project_root/
├── fit_splines.py          # Main pipeline script
├── data/                   # Input folder with .npy point-clouds
│   ├── plant1.npy
│   ├── plant2.npy
│   └── ...
└── output_splines/         # Outputs: JSON and PNG files
    ├── plant1_output.json
    ├── plant1_skeleton.png
    ├── plant1_bsplines.png
    └── plant1_combined.png
```

## Usage

```bash
python point_to_splines.py \
  --input_dir ./data \
  --output_dir ./output_splines \
  --extension .npy \
  --log_level INFO
```

- `--input_dir`: Folder containing `.npy` files.  
- `--output_dir`: Folder to save JSON and images.  
- `--extension`: File extension to process (default: `.npy`).  
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR).

## Output Details

For each input file `plantX.npy`, you’ll get:

- **JSON** (`plantX_output.json`):
  - `center_point`: `[0,0,0]` (root shifted to origin)  
  - `branches`: array of objects, each with:
    - `branch_id`
    - `knots`
    - `coefficients`
    - `degree`
    - `length`
    - `direction`
- **PNG Images**:
  - `plantX_skeleton.png` — the point cloud with skeleton overlay.  
  - `plantX_bsplines.png` — fitted splines and control points.  
  - `plantX_combined.png` — side-by-side skeleton and spline views.

Progress and any warnings/errors will appear on the console based on your chosen `--log_level`.

## Customization

- **Control Points**: In `fit_bspline_with_n_control_points()`, change `n_control` for more or fewer control points.  
- **XY Emphasis**: Adjust `xy_emphasis` to weight horizontal variation during fitting.  
- **Simplification Level**: Modify `target_nodes` in `simplify_skeleton()` to change skeleton detail.  
- **Thresholds**: Tweak jump distance, neighbor radius, and edge-length limits in the code for different datasets.

## Example

```bash
# Debug run on two plants
echo "Running spline fitting..."
python point_to_spline.py --input_dir data --output_dir results --log_level DEBUG
```
# Synthetic Plant Generation Pipeline (gen_spline.py)

## Overview

`gen_spline.py` performs data-driven generation of plant branch structures by:

1. **Extracting** control points from existing JSON outputs of branch splines.  
2. **Learning** a multivariate Gaussian distribution over control points.  
3. **Sampling** synthetic branch sets from that distribution, applying upward/drooping biases.  
4. **Filtering** branches by geometric constraints (curvature, symmetry, drooping).  
5. **Constructing** B-spline curves from control points.  
6. **Visualizing** individual plants in 3D.  
7. **Exporting** generated branch sets to JSON and PNG files.  
8. **Batch Mode**: generating and saving a batch of plants programmatically.

## Features

- **Control Point Extraction**: Read `.json` files and collect control points & spline degree.  
- **Distribution Learning**: Compute mean vector and covariance matrix of control points.  
- **Branch Sampling**: Generate variable branch counts via Gaussian sampling or fixed count.  
- **Bias and Droop**: Apply configurable upward or drooping transforms to new branches.  
- **Geometric Filtering**: Enforce curvature, lateral spread, and symmetry thresholds.  
- **Spline Construction**: Build parametric B-spline curves for plotting.  
- **3D Visualization**: Plot each generated plant with color and style variations.  
- **JSON & PNG Export**: Save detailed branch data and corresponding images.  
- **Configurable CLI**: Command-line arguments control input paths, output directories, and generation parameters.

## Dependencies

- Python 3.7+  
- NumPy  
- SciPy  
- Matplotlib  
- mpl_toolkits (for 3D plotting)  
- collections, glob, argparse (std library)  

Install via:

```bash
pip install numpy scipy matplotlib
```

## Project Layout

```
project_root/
├── gen_spline.py            # Main generation and visualization script
├── output_splines/          # Input JSON from spline fitting
│   ├── plant1_output.json
│   └── ...
└── generated_plants/        # Output directory (created by script)
    ├── json/                # Generated JSON branch sets
    │   ├── plant_20250425_001_30_branches.json
    │   └── ...
    └── png/                 # PNG visualizations
        ├── plant_20250425_001_30_branches.png
        └── ...
```

## Usage

Generate a batch of synthetic plants:

```bash
python gen_spline.py --input output_splines   --output generated_plants   --num_plants 10
```

For more control over plant generation

```bash
python gen_spline.py   --input output_splines   --output generated_plants   --num_plants 10   --upward_bias 0.55   --drooping   --include_upward   --upward_ratio 0.5   --width 12   --height 10   --gaussian   --branch_mean 32   --branch_variance 5   --batch_size 200   --max_attempts 20
```

- `--input`: Path to JSON file or directory of existing branch data.  
- `--output`: Directory where generated JSON/PNG will be saved.  
- `--num_plants`: Number of plant structures to generate.  
- `--upward_bias`: Strength of upward growth bias.  
- `--drooping`: Enable drooping branch generation.  
- `--include_upward`: Also generate non-drooping (upward) branches.  
- `--upward_ratio`: Fraction of branches that grow upward.  
- `--width`, `--height`: Dimensions for saved PNG plots.  
- `--gaussian`: Use Gaussian distribution for branch count.  
- `--branch_mean`, `--branch_variance`: Stats for branch count sampling.  
- `--batch_size`: Branches generated per attempt.  
- `--max_attempts`: Maximum trials to reach target branch count.

## Output

- **JSON (`/generated_plants/json/`)**:  
  - `source_file`: Original JSON or filepath.  
  - `center_point`: Plant base (default `[0,0,0]`).  
  - `branch_count`: Number of branches generated.  
  - `branches`: Array of control-point lists, knots, degree, length, direction.

- **PNG (`/generated_plants/png/`)**:  
  - 3D render of each plant’s spline network.

## Customization

- Adjust drooping logic in `generate_branches()`.  
- Tweak filter thresholds in `filter_branches()`.  
- Change spline detail via `construct_spline_curves()` parameters.  
- Modify sampling distribution with `sample_branch_count()`.

## Example

```bash
# Generate 5 upward-only plants without drooping
python gen_spline.py --input output_splines --output plants_upward   --num_plants 5 --include_upward --drooping False
```
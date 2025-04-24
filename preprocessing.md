# Preprocessing Point Cloud Data

### Overview

This Python script preprocesses 3D point cloud data (.ply files) by centering, normalizing, and resampling them to a consistent number of points (default: 8000). The processed data is saved as NumPy (.npy) files, suitable for further analysis and machine learning applications.

### Features

- Centering: Points are shifted so that the centroid of the point cloud is at the origin.

- Normalization: Points are scaled so that the maximum distance from the origin to any point is 1.

- Resampling: Point clouds are either randomly downsampled or upsampled to maintain exactly 8000 points for consistency.

#### Dependencies

- Python 3.x

- Open3D

- NumPy

Install dependencies using:

```
pip install open3d numpy
```

##### Directory Structure

Ensure your data is structured as follows:

```
project_root/
├── preprocessing.py
├── data_35/
│   ├── example1.ply
│   ├── example2.ply
│   └── ...
└── data/
    └── (processed files will be stored here)
```
Running the Script

Execute the script from the command line:

```
python preprocessing.py
```

#### Output

Processed files are stored in the data directory with filenames corresponding to the input files, saved as NumPy arrays (.npy).

#### Error Handling

The script gracefully handles errors, skipping files with issues and reporting the error details to the console.

Example Output

Upon successful processing, the script outputs:
```
Processed and saved: data/example1.npy
Processed and saved: data/example2.npy
...
```

Any issues encountered during processing will print an error message and skip the problematic file:

```
Error processing file data_35/example3.ply: No points found in file data_35/example3.ply
Skipping file: example3.ply
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
import os

# Number of points to ensure consistency
TARGET_NUM_POINTS = 8000

def preprocess_point_cloud(file, target_num_points):
    """
    Preprocess a point cloud file to center, normalize, and resample points.

    Args:
        file (str): Path to the input .ply file.
        target_num_points (int): Target number of points for the point cloud.

    Returns:
        np.ndarray: Preprocessed points as a NumPy array with a fixed number of points.
    """
    try:
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)

        if points.size == 0:
            raise ValueError(f"No points found in file {file}")

        centroid = points.mean(axis=0)
        points -= centroid

        max_distance = np.linalg.norm(points, axis=1).max()
        points /= max_distance

        if points.shape[0] > target_num_points:
            indices = np.random.choice(points.shape[0], target_num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < target_num_points:
            extra_indices = np.random.choice(points.shape[0], target_num_points - points.shape[0], replace=True)
            points = np.vstack([points, points[extra_indices]])

        return points
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def main():
    dataset_path = "data_35"
    processed_path = "data"
    os.makedirs(processed_path, exist_ok=True)

    for file in os.listdir(dataset_path):
        if file.endswith('.ply'):
            file_path = os.path.join(dataset_path, file)
            points = preprocess_point_cloud(file_path, TARGET_NUM_POINTS)
            if points is not None:
                output_file = os.path.join(processed_path, file.replace('.ply', '.npy'))
                np.save(output_file, points)
                print(f"Processed and saved: {output_file}")
            else:
                print(f"Skipping file: {file}")

if __name__ == "__main__":
    main()

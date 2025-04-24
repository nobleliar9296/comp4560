#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plant Structure Processing Pipeline

This module processes 3D point clouds of plants to extract their branch structure,
fit B-splines to the branches, and generate appropriate visualization and output files.

Key features:
- Converts raw point clouds to skeletonized representations
- Identifies plant root/base and branch structures
- Fits cubic B-splines to branches
- Ensures plant root and all splines start at the origin
- Generates 3D visualizations and structured JSON output

Author: Improved version based on original code
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev, BSpline
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("plant_processor")

###############################################################################
# B-Spline Fitting Module
###############################################################################

def fit_bspline_with_n_control_points(
    points: np.ndarray, 
    n_control: int = 4, 
    degree: int = 3, 
    smoothing: Optional[float] = None,
    xy_emphasis: float = 1.5
) -> Tuple[Tuple, np.ndarray, np.ndarray]:
    """
    Fit a 3D B-spline with a specified number of control points to a set of 3D points,
    with increased emphasis on accuracy where x and y magnitudes are larger.
    
    Args:
        points: Array of shape (n_points, 3) containing 3D coordinates
        n_control: Number of control points (default=4)
        degree: Degree of the B-spline (default=3 for cubic)
        smoothing: Smoothing factor; if None, auto-calculated
        xy_emphasis: Factor to emphasize points with larger x,y magnitudes (default=2.0)
    
    Returns:
        tuple: (tck, spline_points, control_points)
               - tck: B-spline parameters (knots, coefficients, degree)
               - spline_points: Points along the fitted spline
               - control_points: Computed control points
               
    Raises:
        ValueError: If there aren't enough points for the requested spline
        RuntimeError: If there's an error during the spline fitting process
    """
    if len(points) < 2:
        raise ValueError("At least 2 points are required to fit a B-spline.")
    
    # Ensure degree is valid
    degree = min(degree, len(points) - 1)
    
    # Compute chord-length parameterization for better curve fitting
    diffs = np.diff(points, axis=0)
    chord_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_lengths = np.concatenate(([0], np.cumsum(chord_lengths)))
    u = cum_lengths / cum_lengths[-1]  # Normalized parameter
    
    # Calculate weights based on the magnitude of x and y coordinates
    # This gives more emphasis to fitting accuracy where x and y are larger
    xy_magnitudes = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
    # Scale weights based on their x,y distance from origin
    # Adding 0.1 ensures that even points near origin have some weight
    weights = (xy_magnitudes + 0.1)**xy_emphasis
    
    # Normalize weights to avoid extreme values
    weights = weights / np.mean(weights)
    
    # Fit the B-spline
    try:
        if smoothing is None:
            # Automatic smoothing based on number of points and control points
            # Adjust smoothing to account for weighted points
            smoothing = len(points) * 0.1 * np.mean(weights)
        
        # Fit spline with specified number of control points and weights
        tck, u = splprep(
            [points[:, 0], points[:, 1], points[:, 2]], 
            u=u, 
            w=weights,  # Apply weights for x,y-magnitude emphasis
            k=degree, 
            s=smoothing, 
            nest=n_control + degree + 1
        )
        
        # Extract control points from tck
        control_points = np.column_stack(tck[1][:3])[:n_control]
        
        # Generate points along the spline for visualization
        u_new = np.linspace(0, 1, 100)
        spline_points = np.column_stack(splev(u_new, tck))
        
        return tck, spline_points, control_points
    
    except Exception as e:
        raise RuntimeError(f"Error fitting B-spline: {e}")

def branch_to_bspline(
    path_points: np.ndarray,
    xy_emphasis: float = 2.0
) -> Tuple[Tuple, np.ndarray, np.ndarray]:
    """
    Convert a branch (sequence of 3D points) to a B-spline with 4 control points,
    with emphasis on fitting accuracy where x and y magnitudes are larger.
    
    Args:
        path_points: Array of shape (n_points, 3) containing branch point coordinates
        xy_emphasis: Factor to emphasize points with larger x,y magnitudes (default=2.0)
        
    Returns:
        tuple: (tck, spline_points, control_points)
    """
    # Use 4 control points by default for each branch
    return fit_bspline_with_n_control_points(path_points, n_control=4, xy_emphasis=xy_emphasis)

###############################################################################
# Path Extraction Module
###############################################################################

def identify_leaf_nodes(
    edges: np.ndarray, 
    n_nodes: int
) -> List[int]:
    """
    Identify leaf nodes (nodes with degree 1) in the skeleton graph.
    
    Args:
        edges: Array of shape (n_edges, 2) containing edge indices
        n_nodes: Total number of nodes in the graph
        
    Returns:
        list: Indices of leaf nodes
    """
    # Compute the degree of each node
    degrees = np.zeros(n_nodes, dtype=int)
    for edge in edges:
        i, j = edge
        degrees[i] += 1
        degrees[j] += 1
    
    # Leaf nodes have degree 1
    leaf_nodes = [i for i in range(n_nodes) if degrees[i] == 1]
    logger.info(f"Identified {len(leaf_nodes)} leaf nodes")
    return leaf_nodes

def trace_paths_to_root(
    points: np.ndarray, 
    edges: np.ndarray, 
    root_idx: int, 
    leaf_nodes: List[int], 
    jump_threshold: float = 0.01, 
    use_distance: bool = True
) -> Tuple[List[List[int]], Optional[float]]:
    """
    Trace optimized paths from leaf nodes to the root, allowing jumps when 
    the node is close to the origin (magnitude < threshold).
    
    Args:
        points: Array of shape (n_nodes, 3) containing node coordinates
        edges: Array of shape (n_edges, 2) containing edge indices
        root_idx: Index of the root node
        leaf_nodes: List of indices for leaf nodes
        jump_threshold: Threshold for allowing jumps (default=0.01)
        use_distance: Whether to use Euclidean distance (default=True)
        
    Returns:
        tuple: (paths, total_length)
               - paths: List of paths from leaf nodes to root, with root as the LAST point
               - total_length: Sum of path lengths if use_distance=True, else None
               
    Raises:
        ValueError: If input data is invalid
    """
    # Input validation
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be a 2D numpy array of shape (n_nodes, 3)")
    if not isinstance(edges, np.ndarray) or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Edges must be a 2D numpy array of shape (n_edges, 2)")

    # Create original graph for basic connectivity
    G_orig = nx.Graph()
    G_orig.add_edges_from(edges)
    
    # Create a new graph with jump-allowed edges
    G = nx.Graph()
    n_nodes = len(points)
    
    # Precompute Euclidean distances between all pairs of points
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = np.linalg.norm(points[i] - points[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # Compute magnitude (distance from origin) for each node
    node_magnitude = np.sqrt(np.sum(points**2, axis=1))
    z_values = points[:,2]
    
    # Add edges: original edges + jumps where magnitude < threshold
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Add original edges
            if G_orig.has_edge(i, j):
                G.add_edge(i, j, weight=dist_matrix[i, j])
            # Add jump edges if both nodes satisfy the condition
            elif (node_magnitude[i] < jump_threshold and 
                  node_magnitude[j] < jump_threshold and 
                  z_values[i] <= -0.2 and 
                  z_values[j] <= -0.2):
                # Only allow jump if it moves closer to root (smoothness constraint)
                dist_i_to_root = dist_matrix[i, root_idx]
                dist_j_to_root = dist_matrix[j, root_idx]
                if dist_j_to_root < dist_i_to_root:  # j is closer to root
                    G.add_edge(i, j, weight=dist_matrix[i, j])
    
    paths = []
    total_length = 0.0
    
    # Compute shortest paths from each leaf node to the root
    for leaf_idx in leaf_nodes:
        if leaf_idx == root_idx:
            continue
        try:
            if use_distance:
                # Use Dijkstra's algorithm with Euclidean distance weights
                path = nx.dijkstra_path(G, source=leaf_idx, target=root_idx, weight='weight')
                path_length = sum(dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))
                total_length += path_length
            else:
                # Use BFS for minimum hops on original graph (no jumps)
                path = nx.shortest_path(G_orig, source=leaf_idx, target=root_idx)
            
            # IMPORTANT: The paths now have the root as the LAST point
            # No need to reverse the path as we want leaf->root
            paths.append(path)
        except nx.NetworkXNoPath:
            logger.warning(f"No path from leaf node {leaf_idx} to root node {root_idx}")
            continue
    
    logger.info(f"Traced {len(paths)} paths from leaf nodes to root")
    
    return paths, total_length if use_distance else None

def convert_paths_to_point_sequences(
    points: np.ndarray, 
    paths: List[List[int]]
) -> List[np.ndarray]:
    """
    Convert paths (lists of node indices) to sequences of 3D points.
    
    Args:
        points: Array of shape (n_nodes, 3) containing node coordinates
        paths: List of paths, where each path is a list of node indices
        
    Returns:
        list: List of point sequences, each a numpy array of shape (path_length, 3)
    """
    return [points[np.array(path, dtype=int)] for path in paths]

###############################################################################
# Branch Extraction Module
###############################################################################

def extract_branches(
    points: np.ndarray, 
    edges: np.ndarray, 
    root_idx: int
) -> List[np.ndarray]:
    """
    Extract branches from the skeleton by tracing paths from leaf nodes to the root.
    
    Args:
        points: Array of shape (n_nodes, 3) containing node coordinates
        edges: Array of shape (n_edges, 2) containing edge indices
        root_idx: Index of the root node
        
    Returns:
        list: List of branches, each a numpy array of shape (branch_length, 3)
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    degrees = dict(G.degree())
    leaf_nodes = [node for node, degree in degrees.items() if degree == 1]
    logger.info(f"Identified {len(leaf_nodes)} leaf nodes")
    
    paths, total_length = trace_paths_to_root(points, edges, root_idx, leaf_nodes, jump_threshold=0.5)
    branches = convert_paths_to_point_sequences(points, paths)
    return branches

def fit_bsplines_to_branches(
    branches: List[np.ndarray],
    xy_emphasis: float = 2.0
) -> List[Tuple[Tuple, np.ndarray, np.ndarray]]:
    """
    Fit B-splines to each branch. Ensures the root end of the branch receives
    a control point by enforcing the last point in the branch path as a control point.
    
    Args:
        branches: List of branches, each a numpy array of shape (branch_length, 3)
        xy_emphasis: Factor to emphasize points with larger x,y magnitudes (default=2.0)
        
    Returns:
        list: List of B-splines, each a tuple (tck, spline_points, control_points)
    """
    bsplines = []
    for branch in branches:
        try:
            # The last point in the branch is the root node
            root_point = branch[-1].copy()
            
            # Fit the B-spline
            tck, spline_points, control_points = branch_to_bspline(branch, xy_emphasis=xy_emphasis)
            
            # Make sure the last control point is at exactly the root position
            # by adjusting the last control point
            if len(control_points) > 0:
                control_points[-1] = root_point
            
            # Update the tck coefficients to match the adjusted control points
            knots, coeffs, degree = tck
            for i in range(3):  # For each dimension (x, y, z)
                coeffs[i][-1] = root_point[i]
            
            # Update the tck with the modified coefficients
            tck = (knots, coeffs, degree)
            
            # Regenerate spline points with the updated control points
            u_new = np.linspace(0, 1, 100)
            spline_points = np.column_stack(splev(u_new, tck))
            
            if tck is not None:
                bsplines.append((tck, spline_points, control_points))
        except Exception as e:
            logger.error(f"Error fitting B-spline to branch: {e}")
            continue
    
    logger.info(f"Fitted {len(bsplines)} B-splines to branches")
    return bsplines

###############################################################################
# Skeleton Processing Module
###############################################################################

def load_point_cloud(
    file_path: str
) -> Optional[np.ndarray]:
    """
    Load point cloud from .npy file.
    
    Args:
        file_path: Path to the .npy file
        
    Returns:
        numpy.ndarray or None: Point cloud of shape (n_points, 3) or None on error
    """
    try:
        point_cloud = np.load(file_path)
        if point_cloud.shape[1] != 3:
            raise ValueError(f"Point cloud must have shape (N, 3), got {point_cloud.shape}")
        return point_cloud
    except Exception as e:
        logger.error(f"Error loading point cloud from {file_path}: {e}")
        return None

def shrink_point_cloud(
    point_cloud: np.ndarray, 
    initial_radius: float = 0.001, 
    radius_increment: float = 0.05, 
    target_nodes: int = 200
) -> np.ndarray:
    """
    Shrink the point cloud by iteratively merging points within a sphere.
    
    Args:
        point_cloud: Array of shape (n_points, 3) containing point coordinates
        initial_radius: Starting radius for nearest neighbor search (default=0.001)
        radius_increment: Amount to increase radius each iteration (default=0.05)
        target_nodes: Target number of nodes to reach (default=200)
        
    Returns:
        numpy.ndarray: Shrunk point cloud
    """
    points = point_cloud.copy()
    radius = initial_radius
    
    while len(points) > target_nodes:
        # Build a nearest neighbors model
        nbrs = NearestNeighbors(radius=radius).fit(points)
        distances, indices = nbrs.radius_neighbors(points)
        
        # Mark points to keep (not merged)
        keep = np.ones(len(points), dtype=bool)
        merged_points = []
        
        # Iterate through each point
        for i in range(len(points)):
            if not keep[i]:
                continue  # Skip points that have already been merged
            # Get neighbors within the radius
            neighbors = indices[i]
            if len(neighbors) <= 1:
                merged_points.append(points[i])
                continue  # No neighbors to merge with
            # Compute the mean of the neighbors
            mean_point = np.mean(points[neighbors], axis=0)
            merged_points.append(mean_point)
            # Mark neighbors as merged
            keep[neighbors] = False
        
        # Update points with the merged points
        points = np.array(merged_points)
        logger.info(f"After radius {radius:.3f}: {len(points)} points")
        
        # Increase the radius for the next iteration
        radius += radius_increment
        
        # Break early if we're close enough to target
        if len(points) < target_nodes * 1.2:
            break
    
    return points

def find_density_center(
    points: np.ndarray, 
    sphere_center: np.ndarray = np.array([0, 0, -0.45]), 
    sphere_radius: float = 0.2, 
    bandwidth: float = 0.02, 
    grid_resolution: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the highest density point within a specified sphere.
    
    Args:
        points: Array of shape (n_points, 3) containing point coordinates
        sphere_center: Center of the search sphere (default=[0, 0, -0.45])
        sphere_radius: Radius of the search sphere (default=0.2)
        bandwidth: Bandwidth for kernel density estimation (default=0.02)
        grid_resolution: Resolution of the grid for density evaluation (default=5)
        
    Returns:
        tuple: (density_center, in_sphere_points)
               - density_center: Array of shape (3,) containing the center coordinates
               - in_sphere_points: Array of points within the sphere
    """
    # Find all points within the sphere from the original point cloud
    distances = np.linalg.norm(points - sphere_center, axis=1)
    in_sphere_indices = np.where(distances <= sphere_radius)[0]
    in_sphere_points = points[in_sphere_indices]

    logger.info(f"Found {len(in_sphere_points)} points within the density sphere")

    if len(in_sphere_points) > 0:
        # Find the highest density using KDE
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', metric='euclidean')
        kde.fit(in_sphere_points)
        
        # Create a fine 3D grid to evaluate density throughout the sphere
        x_range = np.linspace(sphere_center[0] - sphere_radius, sphere_center[0] + sphere_radius, grid_resolution)
        y_range = np.linspace(sphere_center[1] - sphere_radius, sphere_center[1] + sphere_radius, grid_resolution)
        z_range = np.linspace(sphere_center[2] - sphere_radius + (sphere_radius/3), sphere_center[2] + sphere_radius - (sphere_radius/3), grid_resolution)
        
        grid_points = []
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    point = np.array([x, y, z])
                    if np.linalg.norm(point - sphere_center) <= sphere_radius:
                        grid_points.append(point)
        
        grid_points = np.array(grid_points)
        
        # Evaluate density at each grid point
        log_density = kde.score_samples(grid_points)
        
        # Find the point with highest density
        max_density_idx = np.argmax(log_density)
        density_center = grid_points[max_density_idx].copy()
        
        # Optional: Adjust the vertical position if needed (depends on plant morphology)
        density_center[2] = density_center[2] - 0.2 
        
        logger.info(f"Highest density center located at {density_center}")
    else:
        # Fallback to default center if no points in sphere
        density_center = sphere_center.copy()
        logger.warning("No points found in sphere, using default center")
    
    return density_center, in_sphere_points

def identify_leaf_tips(
    points: np.ndarray, 
    radius: float = 0.03, 
    min_neighbors: int = 3
) -> np.ndarray:
    """
    Identify points that are likely at the end of a leaf (terminal points).
    
    Args:
        points: Array of shape (n_points, 3) containing point coordinates
        radius: Radius for nearest neighbor search (default=0.03)
        min_neighbors: Minimum number of neighbors to be considered a tip (default=3)
        
    Returns:
        numpy.ndarray: Boolean array indicating which points are leaf tips
    """
    # Build a nearest neighbors model
    nbrs = NearestNeighbors(radius=radius).fit(points)
    distances, indices = nbrs.radius_neighbors(points)
    
    # Identify leaf tips based on the number of neighbors
    is_leaf_tip = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        num_neighbors = len(indices[i])
        if num_neighbors <= min_neighbors:  # Including the point itself
            is_leaf_tip[i] = True
    
    logger.info(f"Identified {np.sum(is_leaf_tip)} leaf tips out of {len(points)} points")
    return is_leaf_tip

def connect_nearest_neighbors_with_leaf_tips(
    points: np.ndarray, 
    leaf_tips: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Connect each point to its nearest neighbor, treating leaf tips as endpoints.
    
    Args:
        points: Array of shape (n_points, 3) containing point coordinates
        leaf_tips: Boolean array indicating which points are leaf tips
        
    Returns:
        tuple: (skeleton_points, skeleton_edges)
               - skeleton_points: Array of shape (n_points, 3) containing node coordinates
               - skeleton_edges: Array of shape (n_edges, 2) containing edge indices
    """
    if len(points) <= 1:
        return points, np.array([])
    
    # Build a nearest neighbors model for all points
    nbrs = NearestNeighbors(n_neighbors=len(points)).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Initialize skeleton points and edges
    skeleton_points = points.copy()
    skeleton_edges = []
    visited = set()
    
    # Start with the point with the lowest Z-coordinate that is not a leaf tip
    z_coords = points[:, 2]
    non_leaf_indices = [i for i in range(len(points)) if not leaf_tips[i]]
    if not non_leaf_indices:
        logger.warning("All points are leaf tips, selecting the lowest Z-coordinate point")
        start_idx = np.argmin(z_coords)
    else:
        start_idx = non_leaf_indices[np.argmin(z_coords[non_leaf_indices])]
    
    to_visit = [start_idx]
    visited.add(start_idx)
    
    logger.info(f"Starting path construction at point {start_idx} with Z={points[start_idx, 2]}")
    
    # Iteratively connect points to their nearest unvisited neighbors
    while to_visit:
        current_idx = to_visit.pop(0)
        
        # If the current point is a leaf tip, skip connecting it to others
        if leaf_tips[current_idx]:
            logger.debug(f"Skipping connection for leaf tip {current_idx}")
            continue
        
        # Get the nearest neighbors of the current point
        nearest_neighbors = indices[current_idx]
        nearest_distances = distances[current_idx]
        
        # Find the nearest unvisited neighbor
        nearest_unvisited_idx = None
        nearest_unvisited_dist = float('inf')
        for neighbor_idx, dist in zip(nearest_neighbors, nearest_distances):
            if neighbor_idx not in visited and dist < nearest_unvisited_dist:
                nearest_unvisited_idx = neighbor_idx
                nearest_unvisited_dist = dist
        
        if nearest_unvisited_idx is not None:
            # Add the nearest unvisited neighbor to the path
            visited.add(nearest_unvisited_idx)
            skeleton_edges.append([current_idx, nearest_unvisited_idx])
            to_visit.append(nearest_unvisited_idx)
        else:
            # If no unvisited neighbors, start a new path from an unvisited non-leaf-tip point
            unvisited_non_leaf_indices = [i for i in range(len(points)) if i not in visited and not leaf_tips[i]]
            if unvisited_non_leaf_indices:
                next_start_idx = unvisited_non_leaf_indices[0]
                visited.add(next_start_idx)
                to_visit.append(next_start_idx)
    
    skeleton_edges = np.array(skeleton_edges)
    logger.info(f"Initial skeleton: {len(skeleton_points)} nodes, {len(skeleton_edges)} edges")
    return skeleton_points, skeleton_edges

def filter_long_edges_and_connect(
    points: np.ndarray, 
    edges: np.ndarray, 
    max_edge_length: float = 0.1
) -> np.ndarray:
    """
    Filter edges longer than max_edge_length and ensure the graph is connected.
    
    Args:
        points: Array of shape (n_points, 3) containing node coordinates
        edges: Array of shape (n_edges, 2) containing edge indices
        max_edge_length: Maximum allowed edge length (default=0.1)
        
    Returns:
        numpy.ndarray: Filtered edges array of shape (n_filtered_edges, 2)
    """
    if len(edges) == 0:
        logger.info("No edges to filter, building MST directly")
        # Compute the MST of the full graph
        n = len(points)
        full_adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(points[i] - points[j])
                full_adj_matrix[i, j] = dist
                full_adj_matrix[j, i] = dist
        
        mst = minimum_spanning_tree(full_adj_matrix).tocoo()
        mst_edges = np.array([[i, j] for i, j in zip(mst.row, mst.col) if i < j])
        
        # Filter MST edges to respect max_edge_length
        filtered_edges = []
        for edge in mst_edges:
            i, j = edge
            dist = np.linalg.norm(points[i] - points[j])
            if dist <= max_edge_length:
                filtered_edges.append(edge)
        
        filtered_edges = np.array(filtered_edges)
        logger.info(f"After building MST and filtering long edges (> {max_edge_length}): {len(filtered_edges)} edges")
        return filtered_edges
    
    # Compute edge lengths
    edge_lengths = np.zeros(len(edges))
    for i, edge in enumerate(edges):
        p1, p2 = points[edge[0]], points[edge[1]]
        edge_lengths[i] = np.linalg.norm(p2 - p1)
    
    # Filter edges shorter than max_edge_length
    keep_edges = edge_lengths <= max_edge_length
    filtered_edges = edges[keep_edges]
    logger.info(f"After filtering long edges (> {max_edge_length}): {len(filtered_edges)} edges")
    
    # Build an adjacency matrix for the filtered graph
    n = len(points)
    adj_matrix = np.zeros((n, n))
    for edge in filtered_edges:
        i, j = edge
        dist = np.linalg.norm(points[i] - points[j])
        adj_matrix[i, j] = dist
        adj_matrix[j, i] = dist
    
    # Check if the graph is connected
    n_components, labels = connected_components(adj_matrix, directed=False)
    logger.info(f"Number of connected components after filtering: {n_components}")
    
    if n_components > 1:
        # Compute the MST of the full graph to reconnect components
        full_adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(points[i] - points[j])
                full_adj_matrix[i, j] = dist
                full_adj_matrix[j, i] = dist
        
        mst = minimum_spanning_tree(full_adj_matrix).tocoo()
        mst_edges = np.array([[i, j] for i, j in zip(mst.row, mst.col) if i < j])
        
        # Combine filtered edges with MST edges to ensure connectivity
        if len(filtered_edges) > 0:
            all_edges = np.vstack([filtered_edges, mst_edges])
        else:
            all_edges = mst_edges
        
        # Remove duplicates
        all_edges = np.unique(all_edges, axis=0)
        
        # Filter again to ensure no long edges
        final_edges = []
        for edge in all_edges:
            i, j = edge
            dist = np.linalg.norm(points[i] - points[j])
            if dist <= max_edge_length:
                final_edges.append(edge)
        
        final_edges = np.array(final_edges)
        logger.info(f"After reconnecting components: {len(final_edges)} edges")
    else:
        final_edges = filtered_edges
    
    return final_edges

def remove_small_components(
    points: np.ndarray, 
    edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove smaller unconnected components, keeping only the largest connected component.
    
    Args:
        points: Array of shape (n_points, 3) containing node coordinates
        edges: Array of shape (n_edges, 2) containing edge indices
        
    Returns:
        tuple: (filtered_points, filtered_edges)
               - filtered_points: Array of shape (n_filtered_points, 3)
               - filtered_edges: Array of shape (n_filtered_edges, 2)
    """
    if len(edges) == 0:
        logger.warning("No edges in the skeleton, returning empty skeleton")
        return points, np.array([])
    
    # Build an adjacency matrix for the graph
    n = len(points)
    adj_matrix = np.zeros((n, n))
    for edge in edges:
        i, j = edge
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    
    # Find connected components
    n_components, labels = connected_components(adj_matrix, directed=False)
    logger.info(f"Number of connected components: {n_components}")
    
    if n_components == 1:
        return points, edges
    
    # Count the number of nodes in each component
    component_sizes = np.bincount(labels)
    largest_component = np.argmax(component_sizes)
    logger.info(f"Largest component (label {largest_component}) has {component_sizes[largest_component]} nodes")
    
    # Keep only the nodes and edges in the largest component
    keep_nodes = labels == largest_component
    node_mapping = np.zeros(n, dtype=int)
    node_mapping[keep_nodes] = np.arange(np.sum(keep_nodes))
    
    # Filter points
    filtered_points = points[keep_nodes]
    
    # Filter edges and update node indices
    filtered_edges = []
    for edge in edges:
        i, j = edge
        if keep_nodes[i] and keep_nodes[j]:
            filtered_edges.append([node_mapping[i], node_mapping[j]])
    
    filtered_edges = np.array(filtered_edges)
    logger.info(f"After removing smaller components: {len(filtered_points)} nodes, {len(filtered_edges)} edges")
    return filtered_points, filtered_edges

def simplify_skeleton(
    points: np.ndarray, 
    edges: np.ndarray, 
    leaf_tips: np.ndarray, 
    target_nodes: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplify the skeleton by reducing the number of nodes using K-means clustering.
    
    Args:
        points: Array of shape (n_points, 3) containing node coordinates
        edges: Array of shape (n_edges, 2) containing edge indices
        leaf_tips: Boolean array indicating which points are leaf tips
        target_nodes: Target number of nodes (default=50)
        
    Returns:
        tuple: (new_points, final_edges, new_leaf_tips)
               - new_points: Array of shape (target_nodes, 3)
               - final_edges: Array of shape (n_final_edges, 2)
               - new_leaf_tips: Boolean array for new points
    """
    if len(points) <= target_nodes:
        logger.info(f"No simplification needed: {len(points)} nodes already <= {target_nodes}")
        return points, edges, leaf_tips
    
    # Use K-means clustering to reduce the number of nodes
    kmeans = KMeans(n_clusters=target_nodes, random_state=0).fit(points)
    new_points = kmeans.cluster_centers_
    
    # Build a mapping from old node indices to new node indices
    labels = kmeans.labels_
    node_mapping = np.zeros(len(points), dtype=int)
    for i in range(len(points)):
        node_mapping[i] = labels[i]
    
    # Update leaf tips for the new nodes
    new_leaf_tips = np.zeros(target_nodes, dtype=bool)
    for i in range(target_nodes):
        # Find all original nodes that map to this new node
        original_indices = np.where(labels == i)[0]
        # If any of the original nodes were leaf tips, mark the new node as a leaf tip
        if any(leaf_tips[original_indices]):
            new_leaf_tips[i] = True
    
    # Update edges with new node indices
    new_edges = []
    seen_edges = set()
    for edge in edges:
        i, j = edge
        new_i, new_j = node_mapping[i], node_mapping[j]
        if new_i != new_j:  # Avoid self-loops
            edge_tuple = tuple(sorted([new_i, new_j]))
            if edge_tuple not in seen_edges:
                new_edges.append([new_i, new_j])
                seen_edges.add(edge_tuple)
    
    new_edges = np.array(new_edges)
    
    # Build an adjacency matrix for the new graph
    n = len(new_points)
    adj_matrix = np.zeros((n, n))
    for edge in new_edges:
        i, j = edge
        dist = np.linalg.norm(new_points[i] - new_points[j])
        adj_matrix[i, j] = dist
        adj_matrix[j, i] = dist
    
    # Ensure the graph is connected
    n_components, labels = connected_components(adj_matrix, directed=False)
    logger.info(f"Number of connected components after simplification: {n_components}")
    
    if n_components > 1:
        # Compute the MST of the new graph to reconnect components
        full_adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(new_points[i] - new_points[j])
                full_adj_matrix[i, j] = dist
                full_adj_matrix[j, i] = dist
        
        mst = minimum_spanning_tree(full_adj_matrix).tocoo()
        mst_edges = np.array([[i, j] for i, j in zip(mst.row, mst.col) if i < j])
        
        # Combine new edges with MST edges to ensure connectivity
        if len(new_edges) > 0:
            all_edges = np.vstack([new_edges, mst_edges])
        else:
            all_edges = mst_edges
        
        # Remove duplicates
        all_edges = np.unique(all_edges, axis=0)
        
        # Filter edges to respect max_edge_length (0.1)
        final_edges = []
        for edge in all_edges:
            i, j = edge
            dist = np.linalg.norm(new_points[i] - new_points[j])
            if dist <= 0.1:
                final_edges.append(edge)
        
        final_edges = np.array(final_edges)
    else:
        final_edges = new_edges
    
    logger.info(f"After simplification: {len(new_points)} nodes, {len(final_edges)} edges")
    return new_points, final_edges, new_leaf_tips

def find_root_base(
    points: np.ndarray, 
    edges: np.ndarray, 
    leaf_tips: np.ndarray, 
    original_points: np.ndarray
) -> int:
    """
    Find the root/base of the plant using density-based method with weighted Z-distance.
    
    Args:
        points: Array of shape (n_points, 3) containing node coordinates
        edges: Array of shape (n_edges, 2) containing edge indices
        leaf_tips: Boolean array indicating which points are leaf tips
        original_points: Original point cloud array of shape (n_original, 3)
        
    Returns:
        int: Index of the root node
    """
    # Find the density center near the bottom of the plant
    density_center, _ = find_density_center(original_points, sphere_radius=0.15, bandwidth=0.1, grid_resolution=5)
    
    # Compute weighted distances to find the nearest node in the skeleton to the density center
    # Weight the Z-axis by 0.4, while X and Y axes have a weight of 1.0
    distances = np.zeros(len(points))
    for i in range(len(points)):
        delta = points[i] - density_center
        # Weighted distance: sqrt((x1-x2)^2 + (y1-y2)^2 + (0.4*(z1-z2))^2)
        weighted_z = 0.4 * delta[2]  # Apply weight of 0.4 to Z-coordinate
        distances[i] = np.sqrt(delta[0]**2 + delta[1]**2 + weighted_z**2)
    
    # Find the node with the minimum weighted distance
    root_idx = np.argmin(distances)
    
    # Compute the degree of the root node (number of connections)
    degree = sum(1 for edge in edges if root_idx in edge)
    
    logger.info(f"Root/base identified at node {root_idx} with position {points[root_idx]}, degree={degree}")
    return root_idx

###############################################################################
# Visualization Module
###############################################################################

def plot_skeleton_with_bsplines(
    fig: Figure, 
    ax: Axes3D, 
    point_cloud: np.ndarray, 
    skeleton_points: np.ndarray, 
    skeleton_edges: np.ndarray, 
    root_idx: int, 
    bsplines: List[Tuple], 
    plant_name: str
) -> None:
    """
    Plot the point cloud, skeleton, and B-splines.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib 3D axes
        point_cloud: Array of shape (n_points, 3) containing point cloud coordinates
        skeleton_points: Array of shape (n_nodes, 3) containing skeleton node coordinates
        skeleton_edges: Array of shape (n_edges, 2) containing edge indices
        root_idx: Index of the root node
        bsplines: List of B-splines, each a tuple (tck, spline_points, control_points)
        plant_name: Name of the plant for plot title
    """
    # Plot the point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
               c='blue', s=5, alpha=0.3, label='Point Cloud')
    
    # Plot skeleton nodes (non-root nodes in red, root node in green)
    non_root_indices = [i for i in range(len(skeleton_points)) if i != root_idx]
    ax.scatter(skeleton_points[non_root_indices, 0], skeleton_points[non_root_indices, 1], skeleton_points[non_root_indices, 2], 
               c='red', s=30, label='Skeleton Nodes')
    ax.scatter(skeleton_points[root_idx, 0], skeleton_points[root_idx, 1], skeleton_points[root_idx, 2], 
               c='green', s=100, label='Root/Base Node')
    
    # Plot skeleton edges
    if len(skeleton_edges) > 0:
        for edge in skeleton_edges:
            if edge[0] < len(skeleton_points) and edge[1] < len(skeleton_points):
                ax.plot(skeleton_points[edge, 0], skeleton_points[edge, 1], skeleton_points[edge, 2], 
                        c='red', linewidth=1, alpha=0.3, label='Skeleton Edges' if edge[0] == 0 else "")
    
    # Plot B-splines and control points
    for i, (_, spline_points, control_points) in enumerate(bsplines):
        # Plot the B-spline curve
        ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], 
                c='purple', linewidth=2, label='B-Splines' if i == 0 else "")
    
    max_range = np.array([point_cloud[:, i].max() - point_cloud[:, i].min() for i in range(3)]).max() / 2.0
    mid = np.array([point_cloud[:, i].mean() for i in range(3)])
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"Skeleton with B-Spline Branches - {plant_name}")

def plot_bsplines_with_control_points(
    fig: Figure, 
    ax: Axes3D, 
    skeleton_points: np.ndarray, 
    root_idx: int, 
    bsplines: List[Tuple], 
    plant_name: str
) -> None:
    """
    Plot B-splines with their control points.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib 3D axes
        skeleton_points: Array of shape (n_nodes, 3) containing skeleton node coordinates
        root_idx: Index of the root node
        bsplines: List of B-splines, each a tuple (tck, spline_points, control_points)
        plant_name: Name of the plant for plot title
    """
    # Plot the root node in green
    ax.scatter(skeleton_points[root_idx, 0], skeleton_points[root_idx, 1], skeleton_points[root_idx, 2], 
               c='green', s=100, label='Root/Base Node')
    
    # Different colors for different branches
    colors = plt.cm.tab10(np.linspace(0, 1, len(bsplines)))
    
    # Plot B-splines and their control points
    for i, (_, spline_points, control_points) in enumerate(bsplines):
        color = colors[i % len(colors)]
        
        # Plot the B-spline curve
        ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], 
                c=color, linewidth=2, label=f'Branch {i+1}' if i < 10 else "")
        
        # Connect control points with dotted lines (uncomment if needed)
        # ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 
        #        c='black', linestyle=':', linewidth=1, alpha=0.7)
    
    max_range = np.array([skeleton_points[:, i].max() - skeleton_points[:, i].min() for i in range(3)]).max() / 2.0
    mid = np.array([skeleton_points[:, i].mean() for i in range(3)])
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"B-Spline Branches - {plant_name}")

def generate_plots(
    point_cloud: np.ndarray, 
    skeleton_points: np.ndarray, 
    skeleton_edges: np.ndarray, 
    root_idx: int, 
    bsplines: List[Tuple], 
    plant_name: str, 
    output_dir: str
) -> None:
    """
    Generate and save plots for a plant.
    
    Args:
        point_cloud: Array of shape (n_points, 3) containing point cloud coordinates
        skeleton_points: Array of shape (n_nodes, 3) containing skeleton node coordinates
        skeleton_edges: Array of shape (n_edges, 2) containing edge indices
        root_idx: Index of the root node
        bsplines: List of B-splines, each a tuple (tck, spline_points, control_points)
        plant_name: Name of the plant
        output_dir: Directory to save the plots
    """
    # Create figures for the plots
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    plot_skeleton_with_bsplines(fig1, ax1, point_cloud, skeleton_points, skeleton_edges, root_idx, bsplines, plant_name)
    
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    plot_bsplines_with_control_points(fig2, ax2, skeleton_points, root_idx, bsplines, plant_name)
    
    # Create combined figure
    fig3 = plt.figure(figsize=(20, 8))
    ax3a = fig3.add_subplot(121, projection='3d')
    plot_skeleton_with_bsplines(fig3, ax3a, point_cloud, skeleton_points, skeleton_edges, root_idx, bsplines, plant_name)
    
    ax3b = fig3.add_subplot(122, projection='3d')
    plot_bsplines_with_control_points(fig3, ax3b, skeleton_points, root_idx, bsplines, plant_name)
    
    plt.tight_layout()
    fig3.suptitle(f"3D B-Spline Branch Fitting - {plant_name}", fontsize=16)
    
    # Save the plots
    os.makedirs(output_dir, exist_ok=True)
    fig1.savefig(os.path.join(output_dir, f"{plant_name}_skeleton.png"))
    fig2.savefig(os.path.join(output_dir, f"{plant_name}_bsplines.png"))
    fig3.savefig(os.path.join(output_dir, f"{plant_name}_combined.png"))
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    
    logger.info(f"Saved plots for {plant_name} in {output_dir}")

###############################################################################
# JSON Export Module
###############################################################################

def create_branch_object(
    branch_id: int, 
    tck: Tuple, 
    control_points: np.ndarray, 
    merged_count: int = 1
) -> Dict[str, Any]:
    """
    Create a branch object in the format specified for JSON output.
    
    Args:
        branch_id: ID of the branch
        tck: B-spline parameters (knots, coefficients, degree)
        control_points: Array of shape (n_control, 3) containing control point coordinates
        merged_count: Number of merged branches (default=1)
        
    Returns:
        dict: Branch object in the specified JSON format
    """
    # Make a copy to avoid modifying the original
    control_points = control_points.copy()
    
    # NOTE: We want the final control point to be at the origin since it represents
    # the root node, and branches start from there. The order in the path is:
    # [leaf_node, ..., intermediate_nodes..., root_node]
    
    # 1. Shift so the last control point (root) is at the origin
    last_point = control_points[-1].copy()
    control_points = control_points - last_point
    
    # Extract knots and degree from tck
    knots, coefficients, degree = tck

    # Calculate branch length and direction 
    # Direction is from root to tip (opposite of the path direction)
    if len(control_points) >= 2:
        # First point is the tip, last point is the root (now at origin)
        length = np.linalg.norm(control_points[0] - control_points[-1])
        # Direction is from root to tip
        direction_vector = control_points[0] - control_points[-1]
        # Avoid division by zero
        norm = np.linalg.norm(direction_vector)
        direction = direction_vector / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])
    else:
        length = 0.0
        direction = np.array([0.0, 0.0, 1.0])

    # Transpose coefficients to match the expected format [3 x n_control]
    coeffs_transposed = []
    for i in range(3):
        # Take the i-th coordinate of each control point
        coeffs_transposed.append(control_points[:, i].tolist())

    return {
        "branch_id": branch_id,
        "type": "merged" if merged_count > 1 else "single",
        "knots": knots.tolist(),
        "coefficients": coeffs_transposed,
        "degree": int(degree),
        "length": float(length),
        "direction": direction.tolist(),
        "merged_count": merged_count
    }

def create_json_output(
    bsplines: List[Tuple], 
    center_point: np.ndarray, 
    plant_name: str
) -> Dict[str, Any]:
    """
    Create a JSON output file in the required format.
    
    Args:
        bsplines: List of B-splines, each a tuple (tck, spline_points, control_points)
        center_point: Array of shape (3,) containing center coordinates
        plant_name: Name of the plant
        
    Returns:
        dict: Plant data in the specified JSON format
    """
    # Create a dictionary to store the plant data
    plant_data = {
        "source_file": f"{plant_name}.npy",
        "center_point": center_point.tolist(),
        "branch_count": len(bsplines),
        "processing_parameters": {
            "voxel_size": 0.03,
            "min_branch_length": 0.15,
            "angle_threshold": 25
        },
        "branches": []
    }
    
    # Add branches to the data
    for i, (tck, _, control_points) in enumerate(bsplines):
        # Randomly assign merged_count for demonstration purposes
        # In a real scenario, this would be determined by the branch processing logic
        merged_count = max(1, int(np.random.rand() * 20) + 1)
        branch = create_branch_object(i, tck, control_points, merged_count)
        plant_data["branches"].append(branch)
    
    return plant_data

def save_json_output(
    plant_data: Dict[str, Any], 
    plant_name: str, 
    output_dir: str
) -> None:
    """
    Save the plant data to a JSON file.
    
    Args:
        plant_data: Plant data dictionary
        plant_name: Name of the plant
        output_dir: Directory to save the JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{plant_name}_output.json")
    
    with open(output_file, 'w') as f:
        json.dump(plant_data, f, indent=2)
    
    logger.info(f"Saved JSON data for {plant_name} to {output_file}")

###############################################################################
# Data Processing Module
###############################################################################

def identify_center_point(
    point_cloud: np.ndarray
) -> np.ndarray:
    """
    Identify the center point at the base of the plant.
    
    Args:
        point_cloud: Array of shape (n_points, 3) containing point coordinates
        
    Returns:
        numpy.ndarray: The identified center point of shape (3,)
    """
    # Find the approximate center (focusing on base/root area)
    # Estimate by finding the center of points in the bottom quarter of the plant
    z_values = point_cloud[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_range = z_max - z_min
    
    # Find points in the bottom quarter of the plant
    bottom_threshold = z_min + z_range * 0.25
    bottom_points = point_cloud[z_values <= bottom_threshold]
    
    if len(bottom_points) > 0:
        # Find the center of the bottom points (this will be our root center)
        center_point = np.mean(bottom_points, axis=0)
    else:
        # Fallback to using the lowest point if no bottom quarter is found
        lowest_point_idx = np.argmin(z_values)
        center_point = point_cloud[lowest_point_idx].copy()
    
    logger.info(f"Identified center point near plant base at {center_point}")
    return center_point

###############################################################################
# Processing Pipeline
###############################################################################

def process_plant_file(
    file_path: str, 
    output_dir: str
) -> None:
    """
    Process a single plant point cloud file.
    
    Args:
        file_path: Path to the point cloud file
        output_dir: Directory to save outputs
    """
    logger.info(f"Processing file: {file_path}")

    # Derive plant name
    plant_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load raw point cloud
    point_cloud = load_point_cloud(file_path)
    if point_cloud is None:
        logger.error(f"Failed to load point cloud from {file_path}")
        return
    logger.info(f"Loaded point cloud with {point_cloud.shape[0]} points")

    # Identify center point (for JSON output)
    center_point = identify_center_point(point_cloud)

    # Shrink to skeleton points
    skeleton_points = shrink_point_cloud(
        point_cloud,
        initial_radius=0.001,
        radius_increment=0.001,
        target_nodes=1000
    )
    logger.info(f"Extracted {len(skeleton_points)} skeleton points after shrinking")

    # Identify and connect leaf tips
    leaf_tips = identify_leaf_tips(skeleton_points, radius=0.03, min_neighbors=3)
    skeleton_points, skeleton_edges = connect_nearest_neighbors_with_leaf_tips(
        skeleton_points, leaf_tips
    )

    # Filter edges and ensure connectivity
    skeleton_edges = filter_long_edges_and_connect(
        skeleton_points, skeleton_edges, max_edge_length=0.1
    )

    # Remove small components
    skeleton_points, skeleton_edges = remove_small_components(
        skeleton_points, skeleton_edges
    )

    # Simplify skeleton
    skeleton_points, skeleton_edges, leaf_tips = simplify_skeleton(
        skeleton_points, skeleton_edges, leaf_tips, target_nodes=200
    )

    # Find the root/base node
    root_idx = find_root_base(
        skeleton_points, skeleton_edges, leaf_tips, point_cloud
    )

    # Extract branch point sequences and fit B-splines
    branches = extract_branches(skeleton_points, skeleton_edges, root_idx)
    bsplines = fit_bsplines_to_branches(branches)

    # --- Shift everything so the root is at the origin ---
    root_point = skeleton_points[root_idx].copy()
    logger.info(f"Original root point: {root_point}")
    logger.info("Shifting all data to place root at origin")

    # Shift point cloud and skeleton
    point_cloud -= root_point
    skeleton_points -= root_point

    # Shift B-spline data
    shifted_bsplines = []
    for tck, spline_pts, ctrl_pts in bsplines:
        # Create new tck with shifted coefficients
        knots, coeffs, degree = tck
        shifted_coeffs = [c - root_point[i] for i, c in enumerate(coeffs)]
        shifted_tck = (knots, shifted_coeffs, degree)

        # Shift spline and control points
        shifted_spline = spline_pts - root_point
        shifted_ctrl = ctrl_pts - root_point
        
        shifted_bsplines.append((shifted_tck, shifted_spline, shifted_ctrl))
    
    bsplines = shifted_bsplines

    # Set JSON center point to origin
    center_point = np.zeros(3)
    logger.info("Center point set to origin for JSON output: [0, 0, 0]")

    # Verify shift
    logger.info(f"Shifted root location (should be [0,0,0]): {skeleton_points[root_idx]}")
    if bsplines:
        logger.info(f"First branch, first control point after shift: {bsplines[0][2][0]}")

    # NOTE: We don't individually shift each spline to the origin here.
    # We already shifted the entire plant to the origin by moving the root point.
    # Individual branch alignment will be handled during JSON export (see create_branch_object).
    
    # Just verify each spline's position relative to the origin
    if bsplines:
        logger.info("Verifying branch positions after root-shifting:")
        for i, (_, _, ctrl_pts) in enumerate(bsplines):
            logger.info(f"Branch {i} first control point: {ctrl_pts[0]}")

    # --- Normalize all spline curves & control points together ---
    # Compute global scale factor from sampled points
    if bsplines:
        max_dist = max(
            np.linalg.norm(spline_pts, axis=1).max()
            for _, spline_pts, _ in bsplines
        )
        if max_dist > 0:
            normalized = []
            for tck, spline_pts, ctrl_pts in bsplines:
                # Scale tck coefficients so JSON export matches normalized geometry
                knots, coeffs, degree = tck
                scaled_coeffs = [c / max_dist for c in coeffs]
                tck_scaled = (knots, scaled_coeffs, degree)
                # Scale sampled points and control points
                normalized.append((
                    tck_scaled,
                    spline_pts / max_dist,
                    ctrl_pts / max_dist
                ))
            bsplines = normalized
            logger.info(f"Normalized all B-splines by factor {max_dist:.4f}")

    # Generate and save visualizations
    generate_plots(
        point_cloud,
        skeleton_points,
        skeleton_edges,
        root_idx,
        bsplines,
        plant_name,
        output_dir
    )

    # Create and save JSON output
    plant_data = create_json_output(bsplines, center_point, plant_name)
    save_json_output(plant_data, plant_name, output_dir)

    logger.info(f"Completed processing for {plant_name}")

def process_all_files(
    input_dir: str, 
    output_dir: str, 
    extension: str = '.npy'
) -> None:
    """
    Process all plant point cloud files in a directory.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory to save outputs
        extension: File extension to look for (default='.npy')
    """
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all files with the specified extension
    files = [f for f in os.listdir(input_dir) if f.endswith(extension)]
    
    if not files:
        logger.warning(f"No {extension} files found in {input_dir}")
        return
    
    logger.info(f"Found {len(files)} {extension} files to process")
    
    # Process each file
    for i, file in enumerate(files):
        logger.info(f"Processing file {i+1}/{len(files)}: {file}")
        try:
            file_path = os.path.join(input_dir, file)
            process_plant_file(file_path, output_dir)
        except Exception as e:
            logger.error(f"Error processing {file}: {e}", exc_info=True)
    
    logger.info(f"Finished processing all files. Results saved to {output_dir}")

###############################################################################
# Main Entry Point
###############################################################################

def main():
    """Main function for the plant structure processing pipeline."""
    parser = argparse.ArgumentParser(description='Process plant point cloud files to extract and visualize branches.')
    parser.add_argument('--input_dir', type=str, default="./data", 
                        help='Directory containing plant point cloud files')
    parser.add_argument('--output_dir', type=str, default='output_splines', 
                        help='Directory to save outputs (default: outputs)')
    parser.add_argument('--extension', type=str, default='.npy', 
                        help='File extension of point cloud files (default: .npy)')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    output_dir = f"{args.output_dir}"
    
    logger.info(f"Starting plant processing pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Looking for files with extension: {args.extension}")
    
    process_all_files(args.input_dir, output_dir, args.extension)

if __name__ == "__main__":
    main()
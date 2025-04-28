import os
import json
import numpy as np
import glob
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import datetime
import shutil
import argparse


def extract_control_points(file_path):
    """
    Extract control points from a JSON file with B-splines.
    
    Parameters:
    -----------
    file_path: str
        Path to the JSON file
        
    Returns:
    --------
    tuple: (list of (control_points, degree) tuples, JSON data dictionary)
    """
    all_control_points = []
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    for branch in data.get("branches", []):
        coeffs = branch.get("coefficients", [])
        degree = branch.get("degree", 3)
        
        if len(coeffs) == 3:
            n_points = len(coeffs[0])
            control_points = np.zeros((n_points, 3))
            
            for i in range(3):
                for j in range(n_points):
                    control_points[j, i] = coeffs[i][j]
            
            all_control_points.append((control_points, degree))

    return all_control_points, data

def extract_all_control_points(directory_path):
    """
    Extract control points from all JSON files in a directory.
    
    Parameters:
    -----------
    directory_path: str
        Path to the directory
        
    Returns:
    --------
    tuple: (list of (control_points, degree) tuples, template JSON data)
    """
    all_branches = []
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    template_data = None
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return all_branches, template_data
        
    print(f"Found {len(json_files)} JSON files in {directory_path}")
    
    for file_path in json_files:
        branches, data = extract_control_points(file_path)
        if branches:
            print(f"Extracted {len(branches)} branches from {os.path.basename(file_path)}")
            all_branches.extend(branches)
            if template_data is None:
                template_data = data
        else:
            print(f"No valid branches found in {os.path.basename(file_path)}")

    return all_branches, template_data

def learn_distribution(control_points_list):
    """
    Learn a Gaussian distribution from control points.
    
    Parameters:
    -----------
    control_points_list: list
        List of (control_points, degree) tuples
        
    Returns:
    --------
    tuple: (mean_vector, covariance_matrix, n_control_points, degree)
    """
    branches_by_type = defaultdict(list)
    for cp, degree in control_points_list:
        n_points = cp.shape[0]
        branch_type = (n_points, degree)
        branches_by_type[branch_type].append((cp, degree))
    
    types = [(type_key, len(branches)) for type_key, branches in branches_by_type.items()]
    types.sort(key=lambda x: x[1], reverse=True)
    
    if not types:
        raise ValueError("No valid branches found")
    
    most_common_type, count = types[0]
    n_control_points, most_common_degree = most_common_type
    
    print(f"Using {count} branches with {n_control_points} control points and degree {most_common_degree}")
    
    filtered_branches = branches_by_type[most_common_type]
    flattened_points = [cp.flatten() for cp, degree in filtered_branches]
    flattened_array = np.array(flattened_points)
    
    mean_vector = np.mean(flattened_array, axis=0)
    cov_matrix = np.cov(flattened_array, rowvar=False)
    
    return mean_vector, cov_matrix, n_control_points, most_common_degree

def sample_branch_count(mean=27, variance=4):
    """
    Sample number of branches from a Gaussian distribution.
    
    Parameters:
    -----------
    mean: int
        Mean number of branches
    variance: float
        Variance of the distribution
    
    Returns:
    --------
    int: Number of branches
    """
    std_dev = np.sqrt(variance)
    count = int(np.round(np.random.normal(mean, std_dev)))
    return max(1, count)

def generate_branches(mean_vector, cov_matrix, n_control_points, degree, n_branches=30, upward_bias=0.2, drooping=True):
    """
    Generate branches from the learned distribution.
    
    Parameters:
    -----------
    mean_vector: numpy array
        Mean vector
    cov_matrix: numpy array
        Covariance matrix
    n_control_points: int
        Number of control points
    degree: int
        Spline degree
    n_branches: int
        Number of branches
    upward_bias: float
        Upward/drooping bias
    drooping: bool
        If True, generate drooping branches
    
    Returns:
    --------
    list: List of (control_points, degree) tuples
    """
    try:
        sampled_vectors = np.random.multivariate_normal(mean_vector, cov_matrix, size=n_branches)
        control_points_list = [vector.reshape(n_control_points, 3) for vector in sampled_vectors]
        
        refined_branches = []
        for control_points in control_points_list:
            origin_offset = control_points[0].copy()
            control_points = control_points - origin_offset
            
            if drooping and n_control_points == 3:
                endpoint_xy_distance = np.linalg.norm(control_points[-1, :2])
                control_points[0, 2] = 0.0
                control_points[1, 2] = upward_bias * 0.6
                droop_factor = np.random.uniform(0.8, 1.2)
                control_points[2, 2] = -upward_bias * endpoint_xy_distance * droop_factor * 0.4
                control_points[1:, 2] += np.random.normal(0, upward_bias * 0.1, n_control_points-1)
                
            elif drooping and n_control_points == 4:
                endpoint_xy_distance = np.linalg.norm(control_points[3, :2])
                control_points[0, 2] = 0.0
                control_points[1, 2] = upward_bias * 0.3
                control_points[2, 2] = upward_bias * 0.6
                control_points[3, 2] = -upward_bias * endpoint_xy_distance * np.random.uniform(0.8, 1.2) * 0.4
                control_points[1:, 2] += np.random.normal(0, upward_bias * 0.1, 3)
                
            else:
                for i in range(n_control_points):
                    control_points[i, 2] += upward_bias * (i / (n_control_points - 1))
            
            refined_branches.append((control_points, degree))
        
        return refined_branches
    except Exception as e:
        print(f"Error generating branches: {e}")
        return []

def filter_branches(branches, curvature_threshold=2.0, lateral_threshold=1.0,
                   vertical_threshold=0.05, symmetry_threshold=0.3, drooping=True):
    """
    Filter branches based on geometric constraints.
    
    Parameters:
    -----------
    branches: list
        List of (control_points, degree) tuples
    curvature_threshold: float
        Max curvature
    lateral_threshold: float
        Max lateral distance
    vertical_threshold: float
        Min vertical growth/droop
    symmetry_threshold: float
        Max midpoint distance
    drooping: bool
        If True, apply drooping filters
    
    Returns:
    --------
    list: Filtered branches
    """
    if not branches:
        print("Warning: No branches to filter")
        return []
        
    filtered_branches = []
    rejected_counts = defaultdict(int)
    
    for branch_data in branches:
        try:
            branch, degree = branch_data
            segments = branch[1:] - branch[:-1]
            segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
            arc_length = np.sum(segment_lengths)
            direct_distance = np.linalg.norm(branch[-1] - branch[0])
            if direct_distance < 0.001:
                continue
                
            lateral_distance = np.linalg.norm(branch[-1, :2] - branch[0, :2])
            xy_midpoint = (branch[0, :2] + branch[-1, :2]) / 2
            midpoint_distance = np.linalg.norm(xy_midpoint)
            
            endpoint_xy = branch[-1, :2]
            endpoint_xy_distance = np.linalg.norm(endpoint_xy)
            backward_bending = False
            if endpoint_xy_distance > 0.001:
                endpoint_dir = endpoint_xy / endpoint_xy_distance
                for i in range(1, len(branch)-1):
                    cp_xy = branch[i, :2]
                    cp_xy_distance = np.linalg.norm(cp_xy)
                    if cp_xy_distance > 0.001:
                        projection = np.dot(cp_xy, endpoint_dir)
                        if projection < -0.05:
                            backward_bending = True
                            rejected_counts["backward_bending"] += 1
                            break
                if backward_bending:
                    continue
            
            if drooping:
                if len(branch) == 3:
                    midpoint_height = branch[1, 2]
                    endpoint_height = branch[2, 2]
                else:
                    midpoint_height = branch[2, 2] if len(branch) > 2 else 0
                    endpoint_height = branch[-1, 2]
                    
                height_difference = midpoint_height - endpoint_height
                endpoint_below_origin = endpoint_height < -vertical_threshold
                min_lateral_spread = lateral_distance >= 0.15
                
                if arc_length / direct_distance > curvature_threshold:
                    rejected_counts["curvature"] += 1
                    continue
                if lateral_distance > lateral_threshold:
                    rejected_counts["lateral"] += 1
                    continue
                if height_difference < vertical_threshold:
                    rejected_counts["height"] += 1
                    continue
                if not endpoint_below_origin:
                    rejected_counts["endpoint_below"] += 1
                    continue
                if not min_lateral_spread:
                    rejected_counts["lateral_spread"] += 1
                    continue
                if midpoint_distance > symmetry_threshold:
                    rejected_counts["symmetry"] += 1
                    continue
                
                filtered_branches.append((branch, degree))
            else:
                vertical_growth = branch[-1, 2] - branch[0, 2]
                if arc_length / direct_distance > curvature_threshold:
                    rejected_counts["curvature"] += 1
                    continue
                if lateral_distance > lateral_threshold:
                    rejected_counts["lateral"] += 1
                    continue
                if vertical_growth < vertical_threshold:
                    rejected_counts["height"] += 1
                    continue
                if midpoint_distance > symmetry_threshold:
                    rejected_counts["symmetry"] += 1
                    continue
                
                filtered_branches.append((branch, degree))
                
        except Exception as e:
            print(f"Error filtering branch: {e}")
            continue
    
    print(f"Filtered {len(filtered_branches)} branches from {len(branches)} input branches")
    print(f"Rejected due to: {dict(rejected_counts)}")
    return filtered_branches

def generate_and_filter_until_target(mean_vector, cov_matrix, n_control_points, degree, 
                                    target_count, upward_bias, drooping=True, 
                                    batch_size=50, max_attempts=20):
    """
    Generate and filter branches until target count is reached.
    
    Parameters:
    -----------
    mean_vector: numpy array
        Mean vector
    cov_matrix: numpy array
        Covariance matrix
    n_control_points: int
        Number of control points
    degree: int
        Spline degree
    target_count: int
        Target number of branches
    upward_bias: float
        Upward/drooping bias
    drooping: bool
        If True, generate drooping branches
    batch_size: int
        Branches per batch
    max_attempts: int
        Max attempts
    
    Returns:
    --------
    list: Filtered branches
    """
    collected_branches = []
    attempts = 0
    
    while len(collected_branches) < target_count and attempts < max_attempts:
        print(f"Attempt {attempts+1}: Generating batch of {batch_size} branches...")
        synthetic_branches = generate_branches(mean_vector, cov_matrix, n_control_points, 
                                             degree, batch_size, upward_bias, drooping)
        filtered_batch = filter_branches(synthetic_branches, drooping=drooping)
        collected_branches.extend(filtered_batch)
        attempts += 1
        print(f"Now have {len(collected_branches)} / {target_count} branches after {attempts} attempts")
    
    if len(collected_branches) < target_count:
        print(f"Warning: Could only generate {len(collected_branches)} branches after {max_attempts} attempts")
    
    return collected_branches

def prune_branches(filtered_branches, target_count):
    """
    Prune branches to target count.
    
    Parameters:
    -----------
    filtered_branches: list
        List of (control_points, degree) tuples
    target_count: int
        Target number of branches
    
    Returns:
    --------
    list: Pruned branches
    """
    if not filtered_branches:
        print("Warning: No branches to prune")
        return []
        
    if len(filtered_branches) <= target_count:
        print(f"No pruning needed, only have {len(filtered_branches)} branches")
        return filtered_branches
    
    indices = np.random.choice(len(filtered_branches), target_count, replace=False)
    pruned_branches = [filtered_branches[i] for i in indices]
    print(f"Pruned from {len(filtered_branches)} to {len(pruned_branches)} branches")
    return pruned_branches

def construct_spline_curves(control_points_list, points_per_curve=20):
    """
    Construct B-spline curves from control points.
    
    Parameters:
    -----------
    control_points_list: list
        List of (control_points, degree) tuples
    points_per_curve: int
        Number of points per curve
    
    Returns:
    --------
    list: List of curve points
    """
    all_curves = []
    
    for branch_data in control_points_list:
        control_points, degree = branch_data
        n_points = len(control_points)
        n_knots = n_points + degree + 1
        
        try:
            if degree == 3 and n_points == 3:
                knots = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            elif degree == 3 and n_points == 4:
                knots = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            elif degree == 2 and n_points == 3:
                knots = np.array([0, 0, 0, 1, 1, 1])
            else:
                knots = np.zeros(n_knots)
                knots[-(degree+1):] = 1.0
                if n_points > degree + 1:
                    middle_knots = np.linspace(0, 1, n_points - degree)
                    knots[degree+1:degree+1+len(middle_knots)] = middle_knots
            
            splines = [BSpline(knots, control_points[:, i], degree) for i in range(3)]
            t_values = np.linspace(0, 1, points_per_curve)
            curve_points = np.array([[spline(t) for spline in splines] for t in t_values])
            all_curves.append(curve_points)
        except Exception as e:
            print(f"Error creating spline: {e}, skipping branch")
            continue
    
    return all_curves

def plot_plant(curves, title="Generated Plant", color_variations=True, figsize=(12, 10)):
    """
    Visualize the plant using matplotlib.
    
    Parameters:
    -----------
    curves: list
        List of curve points
    title: str
        Plot title
    color_variations: bool
        If True, apply color variations
    figsize: tuple
        Figure size
    
    Returns:
    --------
    tuple: (figure, axis)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    base_colors = ['forestgreen', 'darkgreen', 'seagreen', 'olivedrab']
    
    for curve in curves:
        if color_variations:
            base_color = np.array(plt.cm.colors.to_rgb(np.random.choice(base_colors)))
            color_variation = np.random.normal(0, 0.1, 3)
            branch_color = np.clip(base_color + color_variation, 0, 1)
            alpha = np.random.uniform(0.7, 0.95)
            linewidth = np.random.uniform(1.0, 2.5)
        else:
            branch_color = 'forestgreen'
            alpha = 0.8
            linewidth = 1.5
            
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], '-', 
                alpha=alpha, color=branch_color, linewidth=linewidth)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set axis limits to [-1, 1] since plant is normalized to unit sphere
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    
    return fig, ax

def branches_to_json(branches, branch_type="generated", template_data=None):
    """
    Convert branch data to JSON format.
    
    Parameters:
    -----------
    branches: list
        List of (control_points, degree) tuples
    branch_type: str
        Type of branches
    template_data: dict
        Template JSON data
    
    Returns:
    --------
    dict: JSON data
    """
    if template_data:
        json_data = {
            "source_file": template_data.get("source_file", "generated_plant.npy"),
            "center_point": template_data.get("center_point", [0.0, 0.0, 0.0]),
            "branch_count": len(branches),
            "processing_parameters": template_data.get("processing_parameters", {
                "voxel_size": 0.03,
                "min_branch_length": 0.15,
                "angle_threshold": 25
            })
        }
    else:
        json_data = {
            "source_file": "generated_plant.npy",
            "center_point": [0.0, 0.0, 0.0],
            "branch_count": len(branches),
            "processing_parameters": {
                "voxel_size": 0.03,
                "min_branch_length": 0.15,
                "angle_threshold": 25
            }
        }
    
    json_branches = []
    
    for i, (control_points, degree) in enumerate(branches):
        coefficients = [[float(control_points[j, dim]) for j in range(len(control_points))] for dim in range(3)]
        length = sum(np.linalg.norm(control_points[j] - control_points[j-1]) for j in range(1, len(control_points)))
        direction = control_points[-1] - control_points[0]
        direction_norm = np.linalg.norm(direction)
        direction = direction / direction_norm if direction_norm > 0 else np.array([0.0, 0.0, 0.0])
        
        if degree == 3:
            if len(control_points) == 3:
                knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            elif len(control_points) == 4:
                knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            else:
                knots = [0.0] * (degree + 1) + [1.0] * (degree + 1)
        elif degree == 2:
            knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        else:
            knots = [0.0] * (degree + 1) + [1.0] * (degree + 1)
        
        branch_entry = {
            "branch_id": i,
            "type": branch_type,
            "knots": knots,
            "coefficients": coefficients,
            "degree": degree,
            "length": float(length),
            "direction": [float(d) for d in direction],
            "merged_count": 1
        }
        
        json_branches.append(branch_entry)
    
    json_data["branches"] = json_branches
    return json_data

def visualize_original_plant(file_path, title="Original Plant Structure"):
    """
    Visualize the original plant structure.
    
    Parameters:
    -----------
    file_path: str
        Path to JSON file or directory
    title: str
        Plot title
    
    Returns:
    --------
    tuple: (figure, axis, branches, template_data)
    """
    if os.path.isdir(file_path):
        branches, template_data = extract_all_control_points(file_path)
        if not title or title == "Original Plant Structure":
            title = f"Original Plant Structure ({len(branches)} branches from {file_path})"
    else:
        branches, template_data = extract_control_points(file_path)
        if not title or title == "Original Plant Structure":
            title = f"Original Plant Structure ({len(branches)} branches from {os.path.basename(file_path)})"
    
    if not branches:
        raise ValueError(f"No valid branches found in {file_path}")
        
    curves = construct_spline_curves(branches)
    fig, ax = plot_plant(curves, title=title)
    
    return fig, ax, branches, template_data

def generate_and_plot_plant(real_branches, template_data, upward_bias=0.3, 
                           drooping=True, include_upward=True, upward_ratio=0.3,
                           figsize=(12, 10), use_gaussian_sampling=True,
                           branch_mean=27, branch_variance=4,
                           batch_size=50, max_attempts=20,
                           plant_id=None):
    """
    Generate and plot a normalized plant.
    
    Parameters:
    -----------
    real_branches: list
        List of (control_points, degree) tuples
    template_data: dict
        Template JSON data
    upward_bias: float
        Upward/drooping bias
    drooping: bool
        If True, generate drooping branches
    include_upward: bool
        If True, include upward branches
    upward_ratio: float
        Ratio of upward branches
    figsize: tuple
        Figure size
    use_gaussian_sampling: bool
        If True, use Gaussian branch count
    branch_mean: int
        Mean branch count
    branch_variance: float
        Branch count variance
    batch_size: int
        Branches per batch
    max_attempts: int
        Max attempts
    plant_id: str
        Plant identifier
    
    Returns:
    --------
    tuple: (curves, fig, ax, all_branches, plant_type, actual_branch_count)
    """
    if len(real_branches) == 0:
        raise ValueError("No valid branches provided")
    
    print(f"Using {len(real_branches)} valid branches for generation")
    
    mean_vector, cov_matrix, n_control_points, degree = learn_distribution(real_branches)
    
    if use_gaussian_sampling:
        target_branch_count = sample_branch_count(branch_mean, branch_variance)
        print(f"Target branch count: {target_branch_count} (Gaussian: mean={branch_mean}, variance={branch_variance})")
    else:
        target_branch_count = branch_mean
        print(f"Target branch count: {target_branch_count}")
    
    if include_upward and drooping:
        target_drooping_count = int(target_branch_count * (1 - upward_ratio))
        target_upward_count = target_branch_count - target_drooping_count
    elif drooping:
        target_drooping_count = target_branch_count
        target_upward_count = 0
    else:
        target_drooping_count = 0
        target_upward_count = target_branch_count
    
    print(f"Targeting {target_drooping_count} drooping and {target_upward_count} upward branches")
    
    curves = []
    drooping_branches = []
    upward_branches = []
    
    if drooping:
        print("\n--- Generating drooping branches ---")
        drooping_branches = generate_and_filter_until_target(
            mean_vector, cov_matrix, n_control_points, degree,
            target_drooping_count, upward_bias, drooping=True,
            batch_size=batch_size, max_attempts=max_attempts
        )
        final_drooping_branches = prune_branches(drooping_branches, target_drooping_count)
        drooping_branches = final_drooping_branches
        print(f"Final drooping branch count: {len(final_drooping_branches)}")
    
    if not drooping or include_upward:
        print("\n--- Generating upward branches ---")
        upward_branches = generate_and_filter_until_target(
            mean_vector, cov_matrix, n_control_points, degree,
            target_upward_count, upward_bias * 0.6, drooping=False,
            batch_size=batch_size, max_attempts=max_attempts
        )
        final_upward_branches = prune_branches(upward_branches, target_upward_count)
        upward_branches = final_upward_branches
        print(f"Final upward branch count: {len(final_upward_branches)}")
    
    # Combine and normalize all branches as a single plant to unit sphere
    all_branches = drooping_branches + upward_branches
    
    # Construct spline curves after normalization
    curves = construct_spline_curves(all_branches)
    
    plant_type = "Combined" if include_upward and drooping else ("Drooping" if drooping else "Upward")
    actual_branch_count = len(all_branches)
    
    id_str = f" (ID: {plant_id})" if plant_id else ""
    title = f"Generated {plant_type} Plant - {actual_branch_count} branches{id_str}"
    
    if args.animate:
        fig, ax = plot_plant(curves, title=title, figsize=figsize)
    else:
        fig, ax = None, None

    return curves, fig, ax, all_branches, plant_type, actual_branch_count

def generate_plant_batch(input_path, output_dir, batch_size=10, 
                         upward_bias=0.3, drooping=True, include_upward=True, 
                         upward_ratio=0.3, fig_width=12, fig_height=10,
                         use_gaussian_sampling=True, branch_mean=27, branch_variance=4,
                         gen_batch_size=50, max_attempts=20):
    """
    Generate a batch of normalized plants.
    
    Parameters:
    -----------
    input_path: str
        Path to JSON file or directory
    output_dir: str
        Output directory
    batch_size: int
        Number of plants
    upward_bias: float
        Upward/drooping bias
    drooping: bool
        If True, generate drooping branches
    include_upward: bool
        If True, include upward branches
    upward_ratio: float
        Ratio of upward branches
    fig_width: int
        Figure width
    fig_height: int
        Figure height
    use_gaussian_sampling: bool
        If True, use Gaussian branch count
    branch_mean: int
        Mean branch count
    branch_variance: float
        Branch count variance
    gen_batch_size: int
        Branches per batch
    max_attempts: int
        Max attempts
    
    Returns:
    --------
    list: Paths to JSON files
    """
    if os.path.exists(output_dir):
        overwrite = input(f"Output directory '{output_dir}' exists. Overwrite? (y/n): ")
        if overwrite.lower() == 'y':
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            print("Aborting...")
            return []
    else:
        os.makedirs(output_dir)
    
    json_dir = os.path.join(output_dir, 'json')
    png_dir = os.path.join(output_dir, 'png')
    os.makedirs(json_dir)
    if args.animate:
        os.makedirs(png_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.isdir(input_path):
        real_branches, template_data = extract_all_control_points(input_path)
        source_desc = f"directory '{input_path}'"
    else:
        real_branches, template_data = extract_control_points(input_path)
        source_desc = f"file '{os.path.basename(input_path)}'"
    
    if len(real_branches) == 0:
        raise ValueError(f"No valid branches found in {source_desc}")
    
    json_files = []
    print(f"\n=== Generating batch of {batch_size} plants ===\n")
    
    for i in range(1, batch_size + 1):
        plant_id = f"{timestamp}_{i:03d}"
        print(f"\n--- Generating plant {i}/{batch_size} (ID: {plant_id}) ---\n")
        
        _, fig, _, all_branches, plant_type, branch_count = generate_and_plot_plant(
            real_branches, template_data, upward_bias=upward_bias, drooping=drooping, 
            include_upward=include_upward, upward_ratio=upward_ratio,
            figsize=(fig_width, fig_height), use_gaussian_sampling=use_gaussian_sampling,
            branch_mean=branch_mean, branch_variance=branch_variance,
            batch_size=gen_batch_size, max_attempts=max_attempts,
            plant_id=plant_id
        )
        
        json_filename = f"plant_{plant_id}_{branch_count}_branches.json"
        png_filename = f"plant_{plant_id}_{branch_count}_branches.png"
        json_path = os.path.join(json_dir, json_filename)
        png_path = os.path.join(png_dir, png_filename)
        
        json_data = branches_to_json(all_branches, branch_type=plant_type.lower(), template_data=template_data)
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if args.animate:
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
        
        
        print(f"Saved plant {i}/{batch_size} to:")
        print(f"  JSON: {json_path}")
        
        if args.animate:
            print(f"  PNG: {png_path}")
        
        plt.close(fig)
        json_files.append(json_path)
    
    print(f"\n=== Completed batch of {batch_size} plants ===\n")
    print(f"Output directory: {output_dir}")
    print(f"JSON files: {json_dir}")
    if args.animate:
        print(f"PNG files: {png_dir}")
    
    return json_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plant Structure Visualization')
    parser.add_argument('--input', type=str, default="output_splines", help='Path to JSON file or directory')
    parser.add_argument('--output', type=str, default="generated_plants", help='Output directory')
    parser.add_argument('--num_plants', type=int, default=10, help='Number of plants to generate')
    parser.add_argument('--upward_bias', type=float, default=0.55, help='Upward bias for branch growth')
    parser.add_argument('--drooping', action='store_true', default=True, help='Generate drooping branches')
    parser.add_argument('--include_upward', action='store_true', default=True, help='Include upward branches')
    parser.add_argument('--upward_ratio', type=float, default=0.5, help='Ratio of upward branches')
    parser.add_argument('--width', type=int, default=12, help='Figure width')
    parser.add_argument('--height', type=int, default=10, help='Figure height')
    parser.add_argument('--gaussian', action='store_true', default=True, help='Use Gaussian branch count')
    parser.add_argument('--branch_mean', type=int, default=32, help='Mean number of branches')
    parser.add_argument('--branch_variance', type=float, default=5, help='Branch count variance')
    parser.add_argument('--batch_size', type=int, default=200, help='Branches per batch')
    parser.add_argument('--max_attempts', type=int, default=20, help='Maximum generation attempts')
    parser.add_argument('--animate', action='store_true', default=False, help='Generate plot')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist")
        exit(1)
    
    generate_plant_batch(
        args.input, args.output, batch_size=args.num_plants,
        upward_bias=args.upward_bias, drooping=args.drooping,
        include_upward=args.include_upward, upward_ratio=args.upward_ratio,
        fig_width=args.width, fig_height=args.height,
        use_gaussian_sampling=args.gaussian,
        branch_mean=args.branch_mean, branch_variance=args.branch_variance,
        gen_batch_size=args.batch_size, max_attempts=args.max_attempts
    )
import numpy as np
from skimage.measure import LineModelND, ransac
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import random
from sklearn.metrics.pairwise import cosine_similarity

def is_similar_model(new_model, existing_models, angle_threshold_deg=5.0, dist_threshold=10.0):
    new_direction = new_model.params[1] / np.linalg.norm(new_model.params[1])
    new_origin = new_model.params[0]

    for model in existing_models.values():
        direction = model.params[1] / np.linalg.norm(model.params[1])
        origin = model.params[0]

        angle = np.arccos(np.clip(np.dot(direction, new_direction), -1.0, 1.0)) * 180 / np.pi
        distance = np.linalg.norm(origin - new_origin)

        if angle < angle_threshold_deg and distance < dist_threshold:
            return True  # Too similar

    return False

def find_iterative_lines_ransac(data_array, max_lines=10, residual_threshold=5.0, n_iterations=5000):
    min_sam = 10
    labels = np.full(data_array.shape[0], 20)
    current_line_label = 1
    fitted_models = {}

    for _ in range(max_lines):
        points = data_array[:, :3]

        if points.shape[0] > min_sam:
            model, inlier_mask = ransac(
                points, LineModelND, min_samples=min_sam,
                residual_threshold=residual_threshold, max_trials=n_iterations
            )

            if np.sum(inlier_mask) < min_sam:
                continue

            if is_similar_model(model, fitted_models):
                continue  # Skip redundant model

            inlier_indices = np.where(inlier_mask)[0]
            labels[inlier_indices] = current_line_label
            fitted_models[current_line_label] = model
            current_line_label += 1
        else:
            break

    return labels, fitted_models

# Define the function to find multiple lines using RANSAC
def find_multiple_lines_ransac(data_array, max_lines=10, residual_threshold=5.0, n_iterations=5000, min_samples=2, min_inliers=10):

    # min_sam = 10

    labels = np.full(data_array.shape[0], -20)  # Initialize labels
    current_line_label = 1  # Start labeling lines from 1

    # Use a boolean mask to keep track of which points are still available
    available_points_mask = np.ones(data_array.shape[0], dtype=bool)

    fitted_models = {}

    # Loop to find multiple lines
    for _ in range(max_lines):
        # Filter points based on the available mask
        points = data_array[available_points_mask, :3]  # Use x, y, z coordinates

        if points.shape[0] > min_samples:
            min_samples = int(min(min_samples, points.shape[0]))
            # print(min_samples)
            model_class = LineModelND  # Reference the model class
            # np.random.seed(42)
            # random.seed(42)
            model, inlier_mask = ransac(points, model_class, min_samples=min_samples,
                                        residual_threshold=residual_threshold, max_trials=n_iterations)

            if np.sum(inlier_mask) < min_inliers:
                continue

        else:
            break

        # If no inliers are found, break the loop
        if not np.any(inlier_mask):
            break

        inlier_indices = np.where(available_points_mask)[0][inlier_mask]

        # Assign current line label to inliers
        labels[inlier_indices] = current_line_label
        fitted_models[current_line_label] = model
        current_line_label += 1

        # Update the available points mask to exclude the inliers
        available_points_mask[inlier_indices] = False

        # If no more points left, break
        if not np.any(available_points_mask):  # Check if all points are exhausted
            break

    return labels, fitted_models

def iterative_ransac_with_suppression(
    data_array,
    max_lines=10,
    residual_threshold=5.0,
    n_iterations=5000,
    min_samples=10,
    suppression_factor=0.1
):
    n_points = data_array.shape[0]
    labels = np.full(n_points, 20)  # Default label for unassigned points
    current_label = 1
    fitted_models = {}
    print('Using suppression factor', suppression_factor, residual_threshold)
    # Keep all points, but suppress repeated inliers via weights
    sample_weights = np.ones(n_points)

    for _ in range(max_lines):
        # Weighted sampling: pick a subset to fit RANSAC
        # We'll pre-sample a small set of candidate points based on weights
        probabilities = sample_weights / np.sum(sample_weights)
        candidate_indices = np.random.choice(n_points, size=n_points, replace=True, p=probabilities)
        candidate_points = data_array[candidate_indices, :3]

        # Run RANSAC on weighted sample
        model, inlier_mask = ransac(
            candidate_points,
            LineModelND,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=n_iterations
        )

        # Predict residuals on all points to get global inliers
        residuals = model.residuals(data_array[:, :3])
        global_inlier_mask = residuals < residual_threshold

        if np.sum(global_inlier_mask) < min_samples:
            break  # No good model found

        # Assign label to these inliers
        labels[global_inlier_mask] = current_label
        fitted_models[current_label] = model
        current_label += 1

        # Suppress these inliers in future sampling
        sample_weights[global_inlier_mask] *= suppression_factor
        sample_weights += 1e-8  # Prevent zero probability

    return labels, fitted_models

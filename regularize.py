import numpy as np
from libraries import DataArray
from scipy.stats import chi2

class Regularize:

    def __init__(self, data_array, threshold=0.1, low_energy_threshold=15, merge_type='p_value', merge_algorithm='gmm', func=None):
        """
        Initialize the Regularize class.

        Parameters:
        - data_array: 2D NumPy array containing data points (columns: x, y, z, ..., cluster labels).
        - threshold: float, p-value threshold for merging clusters.
        """
        self.data = data_array
        self.threshold = threshold
        self.low_energy_threshold = low_energy_threshold
        self.merge_type = merge_type
        self.func = func
        self.merge_algorithm = merge_algorithm

    def calculate_g_matrix_p_value(self, xyz_data, clusters):
        """
        Calculate the G matrix based on Mahalanobis distances between clusters.

        Parameters:
        - xyz_data: 2D NumPy array with x, y, z data for all points.
        - clusters: 1D NumPy array with cluster labels for each point.

        Returns:
        - G: 2D NumPy array (KxK), where K is the number of unique clusters.
        """
        unique_clusters = np.unique(clusters)
        K = len(unique_clusters)

        # Initialize the G matrix
        G = np.zeros((K, K))

        # Calculate means, covariance matrices, and counts for each cluster
        epsilon = 1e-3  # Small factor to penalize clusters with insufficient data
        cluster_stats = {}


        for cluster in unique_clusters:
            cluster_data = xyz_data[clusters == cluster]
            count = cluster_data.shape[0]

            if count > 1:  # Only calculate covariance for sufficient data points
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_cov = np.cov(cluster_data.T)
            else:
                # Handle insufficient data
                cluster_mean = np.mean(cluster_data, axis=0) if count > 0 else np.zeros(xyz_data.shape[1])
                cluster_cov = epsilon * np.eye(xyz_data.shape[1])  # Scale identity matrix

            cluster_stats[cluster] = {
                'mean': cluster_mean,
                'cov': cluster_cov,
                'count': count
            }

        # Fill the G matrix
        for i in range(K):
            for j in range(i + 1, K):  # Only calculate upper triangle
                cluster_i, cluster_j = unique_clusters[i], unique_clusters[j]

                # Use the cluster with more points as the reference
                if cluster_stats[cluster_i]['count'] >= cluster_stats[cluster_j]['count']:
                    ci, cj = cluster_i, cluster_j
                else:
                    ci, cj = cluster_j, cluster_i

                try:
                    # cov = cluster_stats[ci]['cov']
                    # cov = cov + 1e-6 * np.eye(cov.shape[0])
                    # cov_inv = np.linalg.pinv(cov)
                    cov_inv = np.linalg.inv(cluster_stats[ci]['cov'])  # Inverse covariance matrix
                    mean_ci = cluster_stats[ci]['mean']
                    data_cj = xyz_data[clusters == cj]

                    # Calculate Mahalanobis distances for all points in cj relative to ci
                    diff = data_cj - mean_ci
                    mahalanobis_square = np.sum(diff @ cov_inv * diff, axis=1)
                    distances = np.sqrt(np.clip(mahalanobis_square, 0, None))  # Clip negative values to 0
                    avg_distance = np.mean(distances)

                    # Calculate p-value
                    if avg_distance <= 0:
                        p_value = 0
                    else:
                        p_value = 1 - chi2.cdf(avg_distance, df=2)


                    # Apply threshold                    
                    if p_value < self.threshold:
                        p_value = 0

                    G[i, j] = round(p_value, 4)
                    G[j, i] = G[i, j]  # Ensure symmetry

                except np.linalg.LinAlgError:
                    G[i, j] = G[j, i] = 0  # Handle singular covariance matrices gracefully
                    # print(f"Singular covariance matrix for clusters {ci} and {cj}. Setting G[{i},{j}] to 0.")

        # Keep only the maximum value in each column
        for j in range(K):
            non_zero_values = G[:, j][G[:, j] != 0]
            if non_zero_values.size > 0:
                max_value = np.max(non_zero_values)
                # G[:, j][G[:, j] != max_value] = 0
                # G[j, :][G[j, :] != max_value] = 0
                col_indices = np.where(G[:, j] != max_value)[0]
                G[col_indices, j] = 0
                G[j, col_indices] = 0 

        np.fill_diagonal(G, 0)  # Ensure diagonal remains zero
        return G

    def calculate_g_matrix_cdist(self, xyz_data, clusters):
        """
        Calculate the G matrix based on a custom distance metric.

        Parameters:
        - xyz_data: 2D NumPy array with x, y, z data for all points.
        - clusters: 1D NumPy array with cluster labels for each point.
        - get_directions: Function to extract metrics for clusters.

        Returns:
        - G: 2D NumPy array (KxK), where K is the number of unique clusters.
        """

        unique_clusters = np.unique(clusters)
        K = len(unique_clusters)

        # Initialize the G matrix
        G = np.zeros((K, K))

        # Iterate through pairs of clusters to calculate the custom metric
        for i in range(K):
            for j in range(i + 1, K):  # Upper triangle only
                cluster_i, cluster_j = unique_clusters[i], unique_clusters[j]

                # Get data for the two clusters
                track1 = xyz_data[clusters == cluster_i]
                track2 = xyz_data[clusters == cluster_j]

                # Skip if either cluster has no points
                if track1.size == 0 or track2.size == 0:
                    G[i, j] = G[j, i] = 0
                    continue

                # Extract directions and compute the custom metric
                try:
                    end_point1, start_point1, beam_vector1, dirVecTrackNorm1, track_mean1, closest_points1 = self.func(track1)
                    end_point2, start_point2, beam_vector2, dirVecTrackNorm2, track_mean2, closest_points2 = self.func(track2)

                    dist1 = np.linalg.norm(end_point1 - start_point2)
                    dist2 = np.linalg.norm(end_point2 - start_point1)
                    dist3 = np.linalg.norm(end_point2 - end_point1)

                    custom_metric = min(dist1, dist2)                    
                    # Apply threshold
                    if custom_metric > self.low_energy_threshold:
                        custom_metric = 0

                    G[i, j] = round(custom_metric, 4)
                    G[j, i] = G[i, j]  # Ensure symmetry
                except Exception as e:
                    G[i, j] = G[j, i] = 0  # Handle any calculation errors gracefully
                    # print(f"Error calculating metric for clusters {cluster_i} and {cluster_j}: {e}")

        # Retain only the maximum value in each column
        for j in range(K):
            non_zero_values = G[:, j][G[:, j] != 0]
            if non_zero_values.size > 0:
                max_value = np.max(non_zero_values)
                # G[:, j][G[:, j] != max_value] = 0
                # G[j, :][G[j, :] != max_value] = 0
                col_indices = np.where(G[:, j] != max_value)[0]
                G[col_indices, j] = 0
                G[j, col_indices] = 0 

        # Ensure diagonal is zero
        np.fill_diagonal(G, 0)

        return G


    def merge_labels(self):
        """
        Merge clusters based on the G matrix until no further merging is possible.
        Excludes noise points (label -1) during merging but preserves them in the final output.

        Returns:
        - clusters: 1D NumPy array with updated cluster labels.
        """
        xyz_data = self.data[:, DataArray.X.value:DataArray.Z.value + 1]

        if self.merge_type == 'p_value' and self.merge_algorithm == 'gmm':
            # print('pval merge')
            clusters = self.data[:, DataArray.GMM.value].astype(int)
        if self.merge_type == 'cdist' and self.merge_algorithm == 'gmm':
            # print('dist merge')
            clusters = self.data[:, DataArray.merge_p_val.value].astype(int)
        if self.merge_type == 'cdist' and self.merge_algorithm == 'ransac':
            # print('dist merge')
            clusters = self.data[:, DataArray.ransac_labels.value].astype(int)

        # Store noise points for later
        noise_mask = clusters == -1
        noise_points = np.where(noise_mask)[0]
        
        # Work only with non-noise points
        non_noise_mask = clusters != -1
        clusters_non_noise = clusters[non_noise_mask]
        xyz_data_non_noise = xyz_data[non_noise_mask]
        
        unique_clusters = np.unique(clusters_non_noise)

        iteration = 0
        while True:
            iteration += 1
            # print(f"Iteration {iteration}: Unique clusters = {len(unique_clusters)}")

            # Calculate the G matrix only for non-noise points
            if self.merge_type == 'p_value':
                G = self.calculate_g_matrix_p_value(xyz_data_non_noise, clusters_non_noise)
            if self.merge_type == 'cdist':
                G = self.calculate_g_matrix_cdist(xyz_data_non_noise, clusters_non_noise)
            
            # Create mapping of row/column indices to cluster labels
            cluster_label_mapping = {idx: label for idx, label in enumerate(unique_clusters)}
            # Calculate non-zero indices and determine termination
            non_zero_indices = np.argwhere(G != 0)
            if non_zero_indices.size == 0 or iteration > 50:
                # print("No non-zero elements in G matrix. Stopping merge.")
                break

            # Create a mapping for merged clusters
            cluster_mapping = {label: label for label in unique_clusters}

            # Merge clusters based on non-zero elements in G
            for i, j in non_zero_indices:
                if i < j:  # Ensure each pair is processed only once
                    ci, cj = unique_clusters[i], unique_clusters[j]
                    # print(f"Merging clusters {ci} and {cj}.")

                    # Assign the higher cluster label to all points in cj
                    merged_label = max(ci, cj)
                    cluster_mapping[cj] = merged_label
                    cluster_mapping[ci] = merged_label

                    # Update clusters based on the mapping
                    for old_label, new_label in cluster_mapping.items():
                        clusters_non_noise[clusters_non_noise == old_label] = new_label

            # Update cluster labels and unique clusters
            unique_clusters = np.unique(clusters_non_noise)            

        # Reconstruct the full clusters array with noise points preserved
        clusters_final = np.copy(clusters)
        clusters_final[non_noise_mask] = clusters_non_noise
        # Ensure noise points retain their -1 label
        clusters_final[noise_mask] = -1
        
        return clusters_final



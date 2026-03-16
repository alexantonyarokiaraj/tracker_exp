from libraries import DataArray, SCAN, RunParameters, Optimize, VolumeBoundaries
from sklearn.neighbors import NearestNeighbors
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
import ROOT as root
import time
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.decomposition import PCA

# Function to find DBSCAN Clusters
def dbcluster(data_array, N_PROC, nn_neighbor, nn_radius, db_min_samples, sensitivity_, eps_threshold_, eps_mode_):
    """
    Perform DBSCAN clustering on a given data array with adaptive epsilon calculation.

    Parameters:
    - data_array: np.ndarray
        Input data with at least 3 columns (x, y, z).
    - N_PROC: int
        Number of processes for parallel computation.
    - nn_neighbor: int
        Number of nearest neighbors for the NearestNeighbors algorithm.
    - nn_radius: float
        Radius for the NearestNeighbors algorithm.
    - db_min_samples: int
        Minimum samples for a cluster in DBSCAN.
    - sensitivity_: float
        Sensitivity for the KneeLocator.
    - eps_threshold_: float
        Threshold below which epsilon defaults to eps_mode_.
    - eps_mode_: float
        Default epsilon value if calculated epsilon is below threshold.

    Returns:
    - labels_: np.ndarray
        Cluster labels from DBSCAN or [-1, -1] in case of failure.
    - valid_cluster: bool
        True if clustering is successful, False otherwise.
    - epsilon_: float
        The epsilon value used for DBSCAN.
    """
    valid_cluster = True
    epsilon_ = 0  # Default epsilon value
    try:
        # Extract the first three columns (x, y, z)
        extractedData = data_array[:, DataArray.X.value:DataArray.Z.value + 1]

        # Nearest neighbors setup
        neigh = NearestNeighbors(n_neighbors=nn_neighbor)
        nbrs = neigh.fit(extractedData)
        distances, indices = nbrs.kneighbors(extractedData)
        distances = np.sort(distances, axis=0)
        dist_ = distances[:, nn_neighbor-1]

        # KneeLocator to find the optimal epsilon
        kneedle = KneeLocator(
            x=indices[:, 0],
            y=dist_,
            S=sensitivity_,
            curve='convex',
            direction='increasing',
            interp_method='interp1d'
        )
        if kneedle.knee is None:
            raise ValueError("KneeLocator failed to identify a knee point.")

        epsilon_ = round(dist_[int(kneedle.knee)], 2)
        if epsilon_ < eps_threshold_:
            print('EPSILON BELOW THRESHOLD, USING DEFAULT', eps_mode_, epsilon_)
            epsilon_ = eps_mode_

        # DBSCAN clustering
        model = DBSCAN(eps=epsilon_, min_samples=db_min_samples, n_jobs=N_PROC)
        labels_ = model.fit_predict(extractedData)
        return labels_, valid_cluster, epsilon_

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Error: {e}")

    # Return defaults in case of failure
    return np.array([-1, -1]), False, epsilon_


def get_unique_colors(n_colors):
    """
    Generate a list of unique colors for ROOT graphics.
    Uses evenly-spaced HSV hues to guarantee perceptually distinct colors
    for any number of clusters.

    Parameters:
    - n_colors: int, number of unique colors needed

    Returns:
    - colors: list of ROOT TColor indices
    """
    import colorsys
    if n_colors == 0:
        return []
    colors = []
    for i in range(n_colors):
        hue = i / n_colors          # evenly spaced around the colour wheel
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(root.TColor.GetColor(int(r * 255), int(g * 255), int(b * 255)))
    return colors


def plot_3d_projections(data_array, color_column_idx, canvas, pad_positions=[1, 2, 3], filter_label=None):
    """
    Plot 3D projections (XY, YZ, XZ) of data array colored by specified column.
    
    Parameters:
    - data_array: np.ndarray
        Input data with columns [X, Y, Z, Q, DBSCAN, ...].
    - color_column_idx: int or DataArray enum
        Column index to use for coloring (e.g., DataArray.DBSCAN.value).
    - canvas: ROOT.TCanvas
        Canvas object to draw on.
    - pad_positions: list
        Pad positions for XY, YZ, XZ projections (default: [1, 2, 3]).
    - filter_label: int or None
        If specified, only plot points with this label (default: None, plot all labels).
    
    Returns:
    - graphs: dict
        Dictionary containing lists of graph objects for each projection.
    """
    if isinstance(color_column_idx, DataArray):
        color_column_idx = color_column_idx.value
    
    # Get color values
    color_values = data_array[:, color_column_idx]
    unique_labels = np.unique(color_values)
    
    # Generate enough unique colors for all clusters (excluding noise points: -1 and -20)
    non_noise_labels = unique_labels[(unique_labels != -1) & (unique_labels != -20)]
    
    # Filter labels if specified
    if filter_label is not None:
        non_noise_labels = non_noise_labels[non_noise_labels == filter_label]
    
    colors_list = get_unique_colors(len(non_noise_labels))
    
    # Create a mapping from label to color
    label_to_color = {label: colors_list[idx] for idx, label in enumerate(non_noise_labels)}
    
    # Store graph objects to prevent garbage collection
    graphs = {
        'xy': [],
        'yz': [],
        'xz': [],
    }
    
    # XY projection
    canvas.cd(pad_positions[0])
    first_drawn = False
    for label in non_noise_labels:
        mask = color_values == label
        color = label_to_color[label]
        
        points_subset = data_array[mask]
        if len(points_subset) > 0:
            graph_xy = root.TGraph(len(points_subset))
            for i, point in enumerate(points_subset):
                graph_xy.SetPoint(i, point[DataArray.X.value], point[DataArray.Y.value])
            
            graph_xy.SetMarkerColor(color)
            graph_xy.SetMarkerStyle(20)
            graph_xy.SetMarkerSize(0.8)
            if not first_drawn:
                graph_xy.Draw("AP")
                graph_xy.SetTitle("XY Projection")
                graph_xy.GetXaxis().SetTitle("X (mm)")
                graph_xy.GetYaxis().SetTitle("Y (mm)")
                # Set axis limits from RunParameters
                graph_xy.GetXaxis().SetLimits(RunParameters.x_start_bin.value, RunParameters.x_end_bin.value)
                graph_xy.SetMinimum(RunParameters.y_start_bin.value)
                graph_xy.SetMaximum(RunParameters.y_end_bin.value)
                first_drawn = True
            else:
                graph_xy.Draw("P same")
            graphs['xy'].append(graph_xy)
    
    # YZ projection
    canvas.cd(pad_positions[1])
    first_drawn = False
    for label in non_noise_labels:
        mask = color_values == label
        color = label_to_color[label]
        
        points_subset = data_array[mask]
        if len(points_subset) > 0:
            graph_yz = root.TGraph(len(points_subset))
            for i, point in enumerate(points_subset):
                graph_yz.SetPoint(i, point[DataArray.Y.value], point[DataArray.Z.value])
            
            graph_yz.SetMarkerColor(color)
            graph_yz.SetMarkerStyle(20)
            graph_yz.SetMarkerSize(0.8)
            if not first_drawn:
                graph_yz.Draw("AP")
                graph_yz.SetTitle("YZ Projection")
                graph_yz.GetXaxis().SetTitle("Y (mm)")
                graph_yz.GetYaxis().SetTitle("Z (mm)")
                # Set axis limits from RunParameters
                graph_yz.GetXaxis().SetLimits(RunParameters.y_start_bin.value, RunParameters.y_end_bin.value)
                graph_yz.SetMinimum(RunParameters.z_start_bin.value)
                graph_yz.SetMaximum(RunParameters.z_end_bin.value)
                first_drawn = True
            else:
                graph_yz.Draw("P same")
            graphs['yz'].append(graph_yz)
    
    # XZ projection
    canvas.cd(pad_positions[2])
    first_drawn = False
    for label in non_noise_labels:
        mask = color_values == label
        color = label_to_color[label]
        
        points_subset = data_array[mask]
        if len(points_subset) > 0:
            graph_xz = root.TGraph(len(points_subset))
            for i, point in enumerate(points_subset):
                graph_xz.SetPoint(i, point[DataArray.X.value], point[DataArray.Z.value])
            
            graph_xz.SetMarkerColor(color)
            graph_xz.SetMarkerStyle(20)
            graph_xz.SetMarkerSize(0.8)
            if not first_drawn:
                graph_xz.Draw("AP")
                graph_xz.SetTitle("XZ Projection")
                graph_xz.GetXaxis().SetTitle("X (mm)")
                graph_xz.GetYaxis().SetTitle("Z (mm)")
                # Set axis limits from RunParameters
                graph_xz.GetXaxis().SetLimits(RunParameters.x_start_bin.value, RunParameters.x_end_bin.value)
                graph_xz.SetMinimum(RunParameters.z_start_bin.value)
                graph_xz.SetMaximum(RunParameters.z_end_bin.value)
                first_drawn = True
            else:
                graph_xz.Draw("P same")
            graphs['xz'].append(graph_xz)
    
    return graphs


def merge_beam_clusters_by_z_centroid(
    data_array,
    label_column,
    track_type_column,
    z_threshold=Optimize.BEAM_Z_MERGE_THRESHOLD_MM.value,
    noise_labels=(-1,),
    beam_value=0,
    z_column=DataArray.Z,
):
    if isinstance(label_column, DataArray):
        label_column = label_column.value
    if isinstance(track_type_column, DataArray):
        track_type_column = track_type_column.value
    if isinstance(z_column, DataArray):
        z_column = z_column.value

    labels = data_array[:, label_column].astype(int)
    track_types = data_array[:, track_type_column].astype(int)
    merged_labels = np.copy(labels)

    beam_mask = track_types == beam_value
    valid_beam_mask = beam_mask & (~np.isin(labels, noise_labels))
    unique_beam_clusters = np.unique(labels[valid_beam_mask])

    centroids_before = {}
    for cluster_label in unique_beam_clusters:
        mask = valid_beam_mask & (labels == cluster_label)
        if np.any(mask):
            centroids_before[int(cluster_label)] = float(np.mean(data_array[mask, z_column]))

    parent = {int(label): int(label) for label in unique_beam_clusters}

    def find(x):
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        root = ra if ra < rb else rb
        parent[ra] = root
        parent[rb] = root

    unique_list = list(map(int, unique_beam_clusters))
    for i, label1 in enumerate(unique_list):
        for label2 in unique_list[i + 1 :]:
            if abs(centroids_before[label1] - centroids_before[label2]) < z_threshold:
                union(label1, label2)

    merged_mapping = {label: find(label) for label in unique_list}

    for old_label, new_label in merged_mapping.items():
        if old_label != new_label:
            mask = valid_beam_mask & (labels == old_label)
            merged_labels[mask] = new_label

    centroids_after = {}
    merged_unique = np.unique(merged_labels[valid_beam_mask])
    for merged_label in merged_unique:
        mask = valid_beam_mask & (merged_labels == merged_label)
        if np.any(mask):
            centroids_after[int(merged_label)] = float(np.mean(data_array[mask, z_column]))

    return merged_labels, centroids_before, centroids_after, merged_mapping


def fit_beam_tracks_pca_constrained_endpoints(
    data_array,
    label_column,
    track_type_column,
    x_start=0.0,
    x_end=256.0,
    y_fixed=128.0,
    noise_labels=(-1,),
    beam_value=0,
    x_column=DataArray.X,
    z_column=DataArray.Z,
):
    if isinstance(label_column, DataArray):
        label_column = label_column.value
    if isinstance(track_type_column, DataArray):
        track_type_column = track_type_column.value
    if isinstance(x_column, DataArray):
        x_column = x_column.value
    if isinstance(z_column, DataArray):
        z_column = z_column.value

    labels = data_array[:, label_column].astype(int)
    track_types = data_array[:, track_type_column].astype(int)

    beam_mask = track_types == int(beam_value)
    valid_beam_mask = beam_mask & (~np.isin(labels, noise_labels))
    unique_beam_clusters = np.unique(labels[valid_beam_mask])

    endpoints = {}
    for cluster_label in map(int, unique_beam_clusters):
        mask = valid_beam_mask & (labels == cluster_label)
        points = data_array[mask]
        if points.shape[0] < 2:
            continue

        xz = points[:, [x_column, z_column]].astype(float)
        mean_xz = xz.mean(axis=0)
        centered = xz - mean_xz

        cov = np.cov(centered.T, bias=True)
        evals, evecs = np.linalg.eigh(cov)
        direction = evecs[:, int(np.argmax(evals))]

        dx = float(direction[0])
        dz = float(direction[1])
        mx = float(mean_xz[0])
        mz = float(mean_xz[1])

        if abs(dx) < 1e-9:
            z1 = mz
            z2 = mz
        else:
            t1 = (float(x_start) - mx) / dx
            t2 = (float(x_end) - mx) / dx
            z1 = mz + t1 * dz
            z2 = mz + t2 * dz

        endpoints[int(cluster_label)] = (
            (float(x_start), float(y_fixed), float(z1)),
            (float(x_end), float(y_fixed), float(z2)),
        )

    return endpoints


def compute_scattered_track_side(
    data_array,
    label_column,
    track_type_column,
    beam_zone_max,
    noise_labels=(-1,),
    scattered_value=1,
    beam_value=0,
    y_column=DataArray.Y,
):
    if isinstance(label_column, DataArray):
        label_column = label_column.value
    if isinstance(track_type_column, DataArray):
        track_type_column = track_type_column.value
    if isinstance(y_column, DataArray):
        y_column = y_column.value

    labels = data_array[:, label_column].astype(int)
    track_types = data_array[:, track_type_column].astype(int)
    side = np.zeros(len(data_array), dtype=int)

    valid_scattered = (track_types == int(scattered_value)) & (~np.isin(labels, noise_labels))
    unique_scattered_labels = np.unique(labels[valid_scattered])

    for cluster_label in map(int, unique_scattered_labels):
        cluster_mask = valid_scattered & (labels == cluster_label)
        if not np.any(cluster_mask):
            continue
        mean_y = float(np.mean(data_array[cluster_mask, y_column]))
        side_value = 1 if mean_y > float(beam_zone_max) else -1
        side[cluster_mask] = side_value

    # Beam tracks (and noise/unassigned) remain 0
    return side


def relabel_small_clusters_to_noise(
    label_values,
    min_cluster_size,
    noise_labels=(-1, -20),
    output_noise_label=-1,
):
    labels_int = np.asarray(label_values, dtype=int).copy()
    min_cluster_size = int(min_cluster_size)

    for lbl in np.unique(labels_int):
        if int(lbl) in noise_labels:
            continue
        if np.sum(labels_int == lbl) < min_cluster_size:
            labels_int[labels_int == lbl] = int(output_noise_label)

    return labels_int


# Function to do the GMM Fitting
def fit_gmm_with_bic(data, max_components=10, min_components=1):
    """
    Fit Gaussian Mixture Model using BIC to select optimal components.

    Parameters:
    - data (np.ndarray): Input data array with shape (n_samples, 6) where columns are x, y, z, q, true labels, ransac labels, gmm labels.
    - max_components (int): Maximum number of GMM components to evaluate for BIC score.

    Returns:
    - best_labels (np.ndarray): Labels assigned to each data point for the GMM model with the lowest BIC.
    - best_n_components (int): Number of components in the best GMM model according to BIC.
    """
    # Extract features (first 3 columns: x, y, z, q)
    features = data[:, :3]

    best_bic = np.inf
    best_gmm = None
    best_n_components = max(1, min_components)

    for n_components in range(max(1, min_components), max_components + 1):
        n_samples = features.shape[0]
        if n_components > features.shape[0]:
            break  # Exit loop early if n_components exceeds n_samples
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(features)
        bic = gmm.bic(features)
        # print('GMM Iterations->', n_components, gmm.n_iter_, gmm.converged_, gmm.tol)

        # Check if this model has the lowest BIC
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n_components = n_components

    # Fit the best GMM model and predict labels
    best_labels = best_gmm.predict(features)
    responsibilities = best_gmm.predict_proba(features)

    return best_labels, best_n_components, responsibilities

# Function to do GMM clustering for every dbscan cluster
def hierarchical_clustering_with_responsibilities(data_array, max_components=10, dbscan_labels=None):
    """
    Perform DBSCAN clustering and then apply GMM clustering to each DBSCAN cluster,
    computing the responsibility array for all data points.

    Parameters:
    - data_array (np.ndarray): Input data array with at least 3 columns (x, y, z).
    - max_components (int): Maximum number of GMM components to evaluate for BIC.
    - dbscan_labels (np.ndarray or None): Pre-computed DBSCAN labels. If None, DBSCAN is run internally.

    Returns:
    - final_labels (np.ndarray): Combined labels for the entire dataset after hierarchical clustering.
    - dbscan_labels (np.ndarray): Labels from the DBSCAN clustering.
    - final_responsibilities (np.ndarray): Responsibility matrix of shape (n_points, total_gmm_clusters).
    """
    # Step 1: Use provided DBSCAN labels or compute them
    if dbscan_labels is not None:
        elapsed_dbscan = 0.0
        valid_cluster = not (len(dbscan_labels) == 2 and np.all(dbscan_labels == -1))
    else:
        start_dbscan = time.perf_counter()
        dbscan_labels, valid_cluster, epsilon_ = dbcluster(
            data_array,
            SCAN.N_PROC.value,
            SCAN.NN_NEIGHBOR.value,
            SCAN.NN_RADIUS.value,
            SCAN.DB_MIN_SAMPLES.value,
            SCAN.SENSITIVITY.value,
            SCAN.EPS_THRESHOLD.value,
            SCAN.EPS_MODE.value
        )
        elapsed_dbscan = time.perf_counter() - start_dbscan

    if not valid_cluster:
        print("DBSCAN clustering failed.")
        return np.array([-1] * len(data_array)), dbscan_labels, None

    unique_clusters = np.unique(dbscan_labels)
    num_points = len(data_array)

    final_labels = -1 * np.ones(num_points, dtype=int)
    final_responsibilities = -1 * np.ones((num_points, 0))

    current_label_offset = 0
    elapsed_gmm =[]
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue

        cluster_mask = dbscan_labels == cluster_id
        cluster_data = data_array[cluster_mask]

        # Force min components if cluster has points both inside and outside the beam zone
        _y = cluster_data[:, DataArray.Y.value]
        _n_inside  = np.sum((_y >= VolumeBoundaries.BEAM_ZONE_MIN.value) & (_y <= VolumeBoundaries.BEAM_ZONE_MAX.value))
        _n_outside = np.sum((_y <  VolumeBoundaries.BEAM_ZONE_MIN.value) | (_y >  VolumeBoundaries.BEAM_ZONE_MAX.value))
        _min_comp = (Optimize.GMM_SPLIT_MIN_COMPONENTS.value
                     if (_n_inside  >= Optimize.GMM_SPLIT_MIN_BEAM_PTS.value and
                         _n_outside >= Optimize.GMM_SPLIT_MIN_OUTSIDE_PTS.value)
                     else 1)

        start_gmm = time.perf_counter()
        gmm_labels, n_comp, responsibilities = fit_gmm_with_bic(cluster_data, max_components=max_components, min_components=_min_comp)
        end_gmm = time.perf_counter()
        elapsed_gmm.append(end_gmm - start_gmm)
        # print(f"GMM computation time: {end_gmm-start_gmm:.6f} seconds")

        global_gmm_labels = gmm_labels + current_label_offset
        final_labels[cluster_mask] = global_gmm_labels

        new_responsibilities = -1 * np.ones((num_points, n_comp))
        new_responsibilities[cluster_mask, :] = responsibilities
        final_responsibilities = np.hstack((final_responsibilities, new_responsibilities))

        current_label_offset += n_comp

    return final_labels, current_label_offset, final_responsibilities, dbscan_labels, elapsed_dbscan, elapsed_gmm 


# Function to plot the kinematics of GMM Clusters
def get_directions(
    data,
    beam_start=None,
    beam_end=None,
    include_z=True,
    beam_plane_y=VolumeBoundaries.BEAM_CENTER.value,
):
    """Compute a PCA direction for a track and extract (start,end) points.

    - If include_z=True: operate in XYZ (3D).
    - If include_z=False: operate in XY only (2D), effectively ignoring Z.
    - If beam_start/beam_end are missing or degenerate, start/end are selected
        using distance to the XZ beam plane at y=beam_plane_y.

    The function auto-reduces constant dimensions (e.g. constant Y) to avoid PCA issues,
    but returns vectors/points in the requested output dimensionality (2D or 3D).
    """

    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected data with shape (N, >=2), got {data.shape}")

    base_dims = [0, 1, 2] if include_z else [0, 1]
    if max(base_dims) >= data.shape[1]:
        raise ValueError(
            f"include_z={include_z} requires at least {max(base_dims)+1} columns; got {data.shape[1]}"
        )

    data_base = data[:, base_dims]
    stds = np.std(data_base, axis=0)
    active_mask = stds > 0
    if not np.any(active_mask):
        raise ValueError("All selected features appear constant; cannot compute direction.")

    data_active = data_base[:, active_mask]

    if len(np.unique(data_base, axis=0)) < len(data_base):
        print("Warning: Duplicate rows detected.")
    if np.any(np.isnan(data_base)) or np.any(np.isinf(data_base)):
        print("Warning: Data contains NaN or inf values.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if data_active.shape[1] == 1:
            dir_active = np.array([1.0])
        else:
            pca = PCA(n_components=1)
            pca.fit(data_active)
            dir_active = pca.components_[0]

    dir_full = np.zeros(len(base_dims), dtype=float)
    dir_full[active_mask] = dir_active
    dirVecTrackNorm = dir_full / np.linalg.norm(dir_full)

    track_mean = np.mean(data_base, axis=0)
    closest_points = find_closest_points_on_line(data_base, dirVecTrackNorm, track_mean)

    beam_mean = None
    beam_vector = np.zeros(len(base_dims), dtype=float)
    if beam_start is not None and beam_end is not None:
        beam_start = np.asarray(beam_start, dtype=float)
        beam_end = np.asarray(beam_end, dtype=float)
        beam_start_base = beam_start[: len(base_dims)]
        beam_end_base = beam_end[: len(base_dims)]
        candidate_beam_vector = beam_end_base - beam_start_base
        if np.linalg.norm(candidate_beam_vector) > 0:
            beam_vector = candidate_beam_vector
            beam_mean = 0.5 * (beam_start_base + beam_end_base)

    start_point, end_point = start_end_points(
        closest_points,
        beam_mean=beam_mean,
        dirVecBeam=beam_vector,
        beam_plane_y=beam_plane_y,
    )

    # Prefer the point closer to the beam reference plane in Y as the "start"
    dist_start = abs(start_point[1] - beam_plane_y)
    dist_end = abs(end_point[1] - beam_plane_y)
    if dist_end < dist_start:
        start_point, end_point = end_point, start_point

    return end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points

# Function to find closest points on line
def find_closest_points_on_line(data, direction_vector, cluster_mean):
    """
    Finds the closest points on a line passing through cluster_mean and oriented along direction_vector.

    Args:
        data: A NumPy array of shape (n, d) or (d,) containing the track data.
        direction_vector: A NumPy array of shape (d,) representing the direction of the line.
        cluster_mean: A NumPy array of shape (d,) representing the point through which the line passes.

    Returns:
        A NumPy array of shape (n, d) (or (d,) if input was a single point) containing the closest point(s).
    """
    data = np.asarray(data, dtype=float)
    direction_vector = np.asarray(direction_vector, dtype=float)
    cluster_mean = np.asarray(cluster_mean, dtype=float)

    is_single_point = data.ndim == 1
    if is_single_point:
        data = data.reshape(1, -1)

    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    centered_data = data - cluster_mean
    projections = np.dot(centered_data, direction_vector).reshape(-1, 1)
    closest_points = cluster_mean + projections * direction_vector

    if is_single_point:
        return closest_points[0]
    return closest_points

# Function to find the start and the end points
def start_end_points(pca_points, beam_mean=None, dirVecBeam=None, beam_plane_y=VolumeBoundaries.BEAM_CENTER.value):
    """
    Finds the start and end points on the PCA line based on the shortest and longest distances
    to a reference beam geometry.

    Args:
        pca_points: A NumPy array of shape (n, d) containing the closest points on the PCA line.
        beam_mean: A NumPy array of shape (d,) representing a point on the beam line.
            If None (or if dirVecBeam is invalid), distances are measured to the XZ beam plane.
        dirVecBeam: A NumPy array of shape (d,) representing the direction vector of the beam line.
        beam_plane_y: The Y coordinate of the XZ beam plane used when no beam line is available.

    Returns:
        start_point: The point on the PCA line with the shortest distance to the beam reference.
        end_point: The point on the PCA line with the longest distance to the beam reference.
    """
    pca_points = np.asarray(pca_points, dtype=float)
    if pca_points.ndim != 2 or pca_points.shape[0] == 0:
        raise ValueError(f"Expected pca_points with shape (N, d), got {pca_points.shape}")

    use_beam_line = beam_mean is not None and dirVecBeam is not None
    if use_beam_line:
        beam_mean = np.asarray(beam_mean, dtype=float)
        dirVecBeam = np.asarray(dirVecBeam, dtype=float)
        use_beam_line = (
            beam_mean.shape[0] == pca_points.shape[1] and
            dirVecBeam.shape[0] == pca_points.shape[1] and
            np.linalg.norm(dirVecBeam) > 0
        )

    if use_beam_line:
        distances = np.array([
            np.linalg.norm(point - find_closest_points_on_line(point, dirVecBeam, beam_mean))
            for point in pca_points
        ])
    else:
        # Beam-plane fallback: distance to XZ plane at y=beam_plane_y.
        if pca_points.shape[1] < 2:
            raise ValueError("Beam-plane fallback requires points with at least 2 dimensions (x, y).")
        distances = np.abs(pca_points[:, 1] - float(beam_plane_y))

    # Find the index of the point with the smallest and largest distance
    start_index = np.argmin(distances)
    end_index = np.argmax(distances)

    # Return the start and end points based on the distances
    start_point = pca_points[start_index]
    end_point = pca_points[end_index]

    return start_point, end_point


def find_closest_beam_track(
    scattered_points_xyz,
    start_point,
    dirVecTrackNorm,
    beam_endpoints,
    truncation_mm=Optimize.SCATTERED_TRUNCATION_MM.value, 
):
    """
    Find the closest beam track to a scattered track.

    Fits a PCA line using only the scattered-track points within
    `truncation_mm` of `start_point` along the track direction, then
    returns the beam-track label whose 3D line is closest to that fit.

    Also returns the vertex (point on the truncated scattered-track line
    closest to the closest beam line) and the start/end points of the
    truncated PCA fit.

    Args:
        scattered_points_xyz: (N, 3) array of scattered track 3D points.
        start_point: (3,) start point of the scattered track.
        dirVecTrackNorm: (3,) unit direction vector of the scattered track.
        beam_endpoints: dict mapping beam label -> ((x0,y0,z0), (x1,y1,z1)).
        truncation_mm: Only include points within this distance from start_point
            along the track direction (default Optimize.SCATTERED_TRUNCATION_MM).

    Returns:
        closest_label: int label of the closest beam track, or -1 if none.
        closest_dist_mm: float distance (mm) to the closest beam track line.
        vertex: (3,) point on the truncated scattered track closest to the beam
            track line, or None if no beam track found.
        trunc_start: (3,) start of the truncated PCA fit (projected extreme closer
            to the beam reference plane in Y).
        trunc_end: (3,) end of the truncated PCA fit (projected extreme farther
            from the beam reference plane in Y).
    """
    if not beam_endpoints:
        return -1, float('inf'), None, None, None

    scattered_points_xyz = np.asarray(scattered_points_xyz, dtype=float)
    start_point = np.asarray(start_point, dtype=float)
    dirVecTrackNorm = np.asarray(dirVecTrackNorm, dtype=float)
    dirVecTrackNorm = dirVecTrackNorm / np.linalg.norm(dirVecTrackNorm)

    # Ensure dirVecTrackNorm points from start_point toward the cluster centroid
    # (PCA sign is arbitrary; if it points backward, the [0, truncation_mm]
    #  window would open away from the main track body)
    cluster_centroid = scattered_points_xyz.mean(axis=0)
    if np.dot(cluster_centroid - start_point, dirVecTrackNorm) < 0:
        dirVecTrackNorm = -dirVecTrackNorm

    # Determine which side of the beam plane the main cluster lives on
    centroid_y = float(scattered_points_xyz[:, 1].mean())
    bz_min = float(VolumeBoundaries.BEAM_ZONE_MIN.value)
    bz_max = float(VolumeBoundaries.BEAM_ZONE_MAX.value)
    beam_center = float(VolumeBoundaries.BEAM_CENTER.value)
    if centroid_y >= beam_center:
        # Main cluster is above beam zone: reject strays strictly below it
        side_mask = scattered_points_xyz[:, 1] >= bz_min
    else:
        # Main cluster is below beam zone: reject strays strictly above it
        side_mask = scattered_points_xyz[:, 1] <= bz_max

    # Keep only points with projection in [0, truncation_mm] from start_point
    offsets = (scattered_points_xyz - start_point) @ dirVecTrackNorm
    mask = (offsets >= 0) & (offsets <= truncation_mm)
    truncated = scattered_points_xyz[mask & side_mask]
    if truncated.shape[0] < 2:
        truncated = scattered_points_xyz[mask]   # fallback: relax side filter
    if truncated.shape[0] < 2:
        truncated = scattered_points_xyz         # fallback: use all points

    # Fit PCA line through the truncated points
    mean_t = truncated.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=1)
        pca.fit(truncated - mean_t)
    dir_t = pca.components_[0]  # unit direction of truncated scattered track

    # Truncated fit start/end: projected extremes of the truncated point set
    projs = (truncated - mean_t) @ dir_t
    trunc_start = mean_t + float(projs.min()) * dir_t
    trunc_end = mean_t + float(projs.max()) * dir_t
    # Orient so start is closer to the beam plane (Y ~ BEAM_CENTER)
    beam_y = float(VolumeBoundaries.BEAM_CENTER.value)
    if abs(trunc_end[1] - beam_y) < abs(trunc_start[1] - beam_y):
        trunc_start, trunc_end = trunc_end, trunc_start

    # Compute 3D skew-line distance from each beam line to the truncated fit
    closest_label = -1
    closest_dist_mm = float('inf')

    for label, (p0, p1) in beam_endpoints.items():
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d_beam = p1 - p0
        norm_beam = np.linalg.norm(d_beam)
        if norm_beam < 1e-9:
            continue
        d_beam = d_beam / norm_beam

        cross = np.cross(dir_t, d_beam)
        cross_norm = np.linalg.norm(cross)
        diff = mean_t - p0

        if cross_norm < 1e-9:
            # Parallel lines: perpendicular point-to-line distance
            dist = float(np.linalg.norm(diff - np.dot(diff, d_beam) * d_beam))
        else:
            # Skew lines: distance along the common perpendicular
            dist = float(abs(np.dot(diff, cross)) / cross_norm)

        if dist < closest_dist_mm:
            closest_dist_mm = dist
            closest_label = int(label)

    # Compute vertex: point on the truncated scattered track closest to the
    # closest beam track line, using the skew-line closest-approach formula.
    vertex = None
    if closest_label != -1 and closest_label in beam_endpoints:
        p0_v = np.asarray(beam_endpoints[closest_label][0], dtype=float)
        p1_v = np.asarray(beam_endpoints[closest_label][1], dtype=float)
        d_beam_v = p1_v - p0_v
        norm_v = np.linalg.norm(d_beam_v)
        if norm_v > 1e-9:
            d_beam_v = d_beam_v / norm_v
            w = mean_t - p0_v
            b = float(np.dot(dir_t, d_beam_v))
            d_val = float(np.dot(dir_t, w))
            e_val = float(np.dot(d_beam_v, w))
            denom = 1.0 - b * b
            if abs(denom) < 1e-9:
                # Parallel: vertex is foot of perpendicular from p0_v onto scattered track
                s = -d_val
            else:
                s = (b * e_val - d_val) / denom
            vertex = mean_t + s * dir_t

    return closest_label, closest_dist_mm, vertex, trunc_start, trunc_end

def angle_between(v1, v2, event_id=None):
    """
    Calculate the angle (in degrees) between two vectors.

    Parameters:
        v1 (array-like): The first vector.
        v2 (array-like): The second vector.

    Returns:
        float: The angle between the two vectors in degrees.
    """
    # Compute cosine of the angle
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    except ZeroDivisionError:   # Handle zero-length vectors
        if event_id is not None:
            print(f"Warning: Zero-length vector encountered in event {event_id}. Returning angle as 0.")
        return 0.0  
    except RuntimeWarning:
        if event_id is not None:
            print(f"Warning: Invalid value encountered in event {event_id}. Returning angle as 0.")
        return 0.0
    # Ensure cosine is within valid range
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # Return angle in degrees
    return np.degrees(np.arccos(cos_theta))

def calculate_phi_angle(v, beam_v):
    """
    Calculate the angle (in degrees) of a single vector in the YZ plane
    relative to the positive Y-axis.

    Parameters:
        v (array-like): The vector (3D).

    Returns:
        float: The signed angle of the vector in the YZ plane in degrees.
               Returns 400 if the vector has no magnitude in the YZ plane.
    """
    # Project the vector onto the YZ plane (ignore x-component)
    v_yz = np.array([v[1], v[2]])

    # Compute the norm of the projected vector
    norm_v = np.linalg.norm(v_yz)

    # Handle zero-magnitude vector
    if norm_v == 0:
        return 400  # Return 400 for zero magnitude in YZ plane

    # Reference direction is along the positive Y-axis
    ref_vector = np.array([1, 0])  # Positive Y-axis in YZ plane

    # Compute the dot product and angle
    dot_product = np.dot(v_yz, ref_vector)
    cos_theta = dot_product / norm_v
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure valid range for arccos
    angle = np.degrees(np.arccos(cos_theta))

    # Compute the cross product to determine the sign of the angle
    cross_product_z = v_yz[0] * ref_vector[1] - v_yz[1] * ref_vector[0]

    # Determine the sign of the angle
    if cross_product_z < 0:
        angle = -angle  # Negative direction

    return angle
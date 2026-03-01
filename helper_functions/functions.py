from libraries import DataArray, SCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import DBSCAN

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
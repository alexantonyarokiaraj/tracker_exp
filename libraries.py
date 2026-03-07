from enum import Enum

class DataArray(Enum):
    X = 0
    Y = 1
    Z = 2
    Q = 3   
    DBSCAN = 4
    GMM = 5
    REGULARIZED = 6
    RANSAC = 7
class RunParameters(Enum):
    lib_path = '/home2/user/u0100486/linux/doctorate/github/tracker_exp/lib/'
    files_path = '/home2/user/u0100486/linux/doctorate/DATA/EXPERIMENTAL/e780/'
    x_conversion_factor = 2.0  # units[mm]
    y_conversion_factor = 2.0  # units[mm]
    time_per_sample = 0.08  # units [us]
    drift_velocity_volume = 1.16  # units [cm/us]
    z_conversion_factor = time_per_sample * drift_velocity_volume * 10  # units[mm]
    nbins_x = 128
    x_start_bin = 0 * x_conversion_factor
    x_end_bin = 128 * x_conversion_factor
    nbins_y = 128
    y_start_bin = 0 * y_conversion_factor
    y_end_bin = 128 * y_conversion_factor
    nbins_z = 512
    z_start_bin = 0 * z_conversion_factor
    z_end_bin = 512 * z_conversion_factor
    lookup_table = "LT_GANIL_NewCF_marine.dat"
    missing_pads_info = "HitResponses.dat"    
    NB_COBO = 16
    NB_ASAD = 4
    NB_AGET = 4
    NB_CHANNEL = 68
class SCAN(Enum):
    N_PROC = 1
    NN_NEIGHBOR = 6
    NN_RADIUS = 20.0
    DB_MIN_SAMPLES = 6
    SENSITIVITY = 3
    EPS_THRESHOLD = 4.0
    EPS_MODE = 7.0

class Optimize(Enum):
    P_VALUE = 0.1
    LOW_ENERGY_THRESHOLD = 15

class RansacParameters(Enum):
    MAX_LINES = 100
    RESIDUAL_THRESHOLD = 5.0
    N_ITERATIONS = 5000
    MIN_SAMPLES = 2
    MIN_INLIERS = 10